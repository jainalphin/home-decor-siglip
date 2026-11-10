# # Pix3D: build the dataset and save it to the data folder
#
# Real photos of furniture, each linked to the exact 3D model (mostly IKEA) it shows. Rows are object crops;
# `group_id` is the 3D model, so two photos of the same model are a "same item" pair.
# License: CC BY 4.0 (Sun et al., "Pix3D", CVPR 2018). Confirm on http://pix3d.csail.mit.edu before release.
#
# Step 1 of 3: build only. No HF token or GPU needed. The leak check (`run_leak_check`) runs afterwards on
# the stored copy, and the HF upload comes last.
#
# Output, under one run folder in the data folder: `train/` and `validation/` JSONL shards and `manifest.json`.
# Each row has the crop (`image_b64`, JPEG) and a small copy of the full photo (`source_thumb_b64`, used only
# by the leak check, since a full room photo could match an STL scene).


import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "finetuning")]
from fsutil import fs, show_html   # file helpers and HTML review pages
# %% ---------------------------------------------------------------------------

# 1. Settings
import base64, hashlib, io, json, re, zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image, ImageOps

DATA_ROOT = "decore"
SOURCE = "pix3d"
URL = "http://pix3d.csail.mit.edu/data/pix3d.zip"
WORK = Path("/local_disk0/decore" if Path("/local_disk0").exists() else "/tmp/decore") / SOURCE   # scratch
PAD = 0.10                 # same padding as the STL query crops
MIN_SIDE = 64
DROP_OCCLUDED = True       # drop crops flagged truncated / occluded
VAL_FRACTION = 0.10        # by 3D model, so no model is in both splits
SHARD_ROWS = 100           # rows per JSONL file on disk
JPEG_QUALITY = 95          # crops are re-encoded once; no resize
THUMB_SIDE = 448           # full-photo copy for the leak check only
LICENSE = "CC-BY-4.0"
ATTRIBUTION = "Pix3D (Sun et al., CVPR 2018), http://pix3d.csail.mit.edu"
# Pix3D category -> (category, category_coarse) in the home-decore taxonomy. "misc" and "tool" are dropped.
CATEGORIES = {"bed": ("bed", "bed"), "bookcase": ("bookcase", "storage furniture"),
              "chair": ("chair", "chair or stool"), "desk": ("desk", "table"), "sofa": ("sofa", "sofa"),
              "table": ("table", "table"), "wardrobe": ("wardrobe", "storage furniture")}

Image.MAX_IMAGE_PIXELS = None
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
run_out = f"{DATA_ROOT}/sources/{SOURCE}/{RUN_ID}"
print("local scratch:", WORK, "| run folder:", run_out)

# %% ---------------------------------------------------------------------------

# 2. Write preflight: write one shard-sized text file, read it back, delete it.
# Shards are large text files; this proves fs.put / head handle that size before any real work.
probe = "x" * 50_000_000
probe_path = f"{run_out}/_preflight.txt"
assert fs.put(probe_path, probe, overwrite=True)
size = fs.ls(probe_path)[0].size
same = fs.head(probe_path, size) == probe
fs.rm(probe_path)
print(f"wrote {size:,} bytes, read back identical: {same}")
assert same, "Large text round trip failed; lower SHARD_ROWS and the probe size"

# %% ---------------------------------------------------------------------------

# 3. Download Pix3D to local scratch and build crop rows
zip_path = WORK / "pix3d.zip"
if not zip_path.exists():
    WORK.mkdir(parents=True, exist_ok=True)
    with requests.get(URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(zip_path.with_suffix(".part"), "wb") as f:
            for chunk in r.iter_content(1 << 22):
                f.write(chunk)
    zip_path.with_suffix(".part").rename(zip_path)
if not (WORK / "raw").exists():
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(WORK / "raw")
root = next((WORK / "raw").rglob("pix3d.json")).parent
records = json.loads((root / "pix3d.json").read_text())
print(f"{len(records)} annotations; first record keys: {sorted(records[0])}")


def title(model_path, category):
    # "model/bed/IKEA_MALM_2/model.obj" -> "malm bed"; non-IKEA models get the category only.
    name = Path(model_path).parent.name
    if not name.upper().startswith("IKEA_"):
        return category
    name = re.sub(r"_\d+$", "", name[5:]).replace("_", " ").lower()
    return f"{name} {category}"


def split_of(group):
    value = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "validation" if value < VAL_FRACTION else "train"


rows, skipped = [], {}
for n, r in enumerate(records):
    if r["category"] not in CATEGORIES:
        skipped["category"] = skipped.get("category", 0) + 1
        continue
    if DROP_OCCLUDED and (r.get("truncated") or r.get("occluded")):
        skipped["occluded"] = skipped.get("occluded", 0) + 1
        continue
    x0, y0, x1, y1 = r["bbox"]
    if min(x1 - x0, y1 - y0) < MIN_SIDE:
        skipped["small"] = skipped.get("small", 0) + 1
        continue
    category, coarse = CATEGORIES[r["category"]]
    group = "pix3d:" + str(Path(r["model"]).parent.relative_to("model"))
    rows.append({"image_id": f"pix3d_{n:05d}", "group_id": group, "category": category,
                 "category_coarse": coarse, "title": title(r["model"], category),
                 "source_image": r["img"], "bbox": [int(v) for v in r["bbox"]],
                 "license": LICENSE, "attribution": ATTRIBUTION, "split": split_of(group)})
print(f"kept {len(rows)} crops from {len({x['group_id'] for x in rows})} models; skipped {skipped}")

# %% ---------------------------------------------------------------------------

# 4. Save to the data folder: JSONL shards per split, plus a manifest


def jpeg_b64(image, quality):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode()


def encode(row):
    photo = ImageOps.exif_transpose(Image.open(root / row["source_image"])).convert("RGB")
    x0, y0, x1, y1 = row["bbox"]
    px, py = (x1 - x0) * PAD, (y1 - y0) * PAD
    crop = photo.crop((max(0, x0 - px), max(0, y0 - py), min(photo.width, x1 + px), min(photo.height, y1 + py)))
    photo.thumbnail((THUMB_SIDE, THUMB_SIDE))
    out = {k: v for k, v in row.items() if k != "split"}
    return out | {"image_b64": jpeg_b64(crop, JPEG_QUALITY), "source_thumb_b64": jpeg_b64(photo, 90)}


counts, shards = {}, {}
for split in ("train", "validation"):
    part = [r for r in rows if r["split"] == split]
    counts[split] = {"rows": len(part), "models": len({r["group_id"] for r in part})}
    shards[split] = []
    for start in range(0, len(part), SHARD_ROWS):
        text = "\n".join(json.dumps(encode(r)) for r in part[start:start + SHARD_ROWS]) + "\n"
        path = f"{run_out}/{split}/part-{len(shards[split]):05d}.jsonl"
        assert fs.put(path, text, overwrite=True), path
        shards[split].append(path)
    print(split, counts[split], f"{len(shards[split])} shards")

manifest = {
    "source": SOURCE, "run_id": RUN_ID, "run_out": run_out,
    "format": "JSONL; one row per crop; crop as base64 JPEG in `image_b64`; "
              "full photo (<= 448 px) in `source_thumb_b64` for the leak check only",
    "columns": ["image_id", "group_id", "category", "category_coarse", "title", "source_image", "bbox",
                "license", "attribution", "image_b64", "source_thumb_b64"],
    "counts": counts, "shards": shards, "license": LICENSE, "attribution": ATTRIBUTION, "url": URL,
    "settings": {"pad": PAD, "min_side": MIN_SIDE, "drop_occluded": DROP_OCCLUDED,
                 "val_fraction": VAL_FRACTION, "jpeg_quality": JPEG_QUALITY, "skipped": skipped},
}
assert fs.put(f"{run_out}/manifest.json", json.dumps(manifest, indent=2), overwrite=True)
print("saved:", run_out)
print([f.path for f in fs.ls(run_out)])
