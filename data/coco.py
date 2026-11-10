# # COCO: build the room-photo dataset and save it to the data folder
#
# Real room photos with 5 human captions each (MS COCO). Same selection as `data/v2/coco_home.py`: a home
# object must cover >= 5% of the photo, and only plain CC BY Flickr photos are kept (NC / SA / ND dropped).
# Captions are CC BY 4.0. Teaches text/style on real rooms (like `deepfurniture_scenes`, but photos).
#
# Step 1 of 3: build only. No HF token or GPU needed. The leak check (`run_leak_check`) runs afterwards on
# the stored copy, and the HF upload comes last.
#
# Output, under one run folder in the data folder: `train/` and `validation/` JSONL shards (image = the original COCO JPEG
# as base64) and `manifest.json`.


import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "finetuning")]
from fsutil import fs, show_html   # file helpers and HTML review pages
# %% ---------------------------------------------------------------------------

# 1. Settings
import base64, hashlib, json, re, zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

DATA_ROOT = "decore"
SOURCE = "coco_scenes"
WORK = Path("/local_disk0/decore" if Path("/local_disk0").exists() else "/tmp/decore") / SOURCE   # scratch
ANNOTATIONS = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"   # ~250 MB
SPLITS = ["train2017", "val2017"]
MIN_AREA_FRAC = 0.05       # the largest home object must cover >= 5% of the photo
VAL_FRACTION = 0.05
SHARD_ROWS = 200           # rows per JSONL file on disk (COCO photos are <= 640 px, ~160 KB)
# COCO class -> (category, category_coarse) in the home-decore taxonomy
CATEGORIES = {"couch": ("sofa", "sofa"), "chair": ("chair", "chair or stool"), "bed": ("bed", "bed"),
              "dining table": ("table", "table"), "potted plant": ("potted plant", "potted plant"),
              "vase": ("vase", "decor accessory"), "clock": ("clock", "decor accessory")}

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
run_out = f"{DATA_ROOT}/sources/{SOURCE}/{RUN_ID}"
print("local scratch:", WORK, "| run folder:", run_out)

# %% ---------------------------------------------------------------------------

# 2. Write preflight: write one shard-sized text file, read it back, delete it.
probe = "x" * 50_000_000
probe_path = f"{run_out}/_preflight.txt"
assert fs.put(probe_path, probe, overwrite=True)
size = fs.ls(probe_path)[0].size
same = fs.head(probe_path, size) == probe
fs.rm(probe_path)
print(f"wrote {size:,} bytes, read back identical: {same}")
assert same, "Large text round trip failed; lower SHARD_ROWS and the probe size"

# %% ---------------------------------------------------------------------------

# 3. Select CC BY room photos from the annotations and download them to local scratch


def fetch(url, path):
    if path.exists():
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
        return True
    except requests.RequestException:
        return False


def cc_by(url):
    # Only plain CC BY passes: "/by-nc/", "/by-sa/", "/by-nd/" do not match "/by/".
    m = re.search(r"creativecommons\.org/licenses/by/([\d.]+)", str(url).lower())
    return f"CC-BY-{m.group(1)}" if m else None


def split_of(image_id):
    value = int(hashlib.sha256(image_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "validation" if value < VAL_FRACTION else "train"


zip_path = WORK / "annotations_trainval2017.zip"
assert fetch(ANNOTATIONS, zip_path), f"could not download {ANNOTATIONS}"

rows = []
with zipfile.ZipFile(zip_path) as z:
    for split in SPLITS:
        inst = json.loads(z.read(f"annotations/instances_{split}.json"))
        caps = json.loads(z.read(f"annotations/captions_{split}.json"))
        licenses = {l["id"]: l["url"] for l in inst["licenses"]}
        names = {c["id"]: c["name"] for c in inst["categories"]}
        images = {im["id"]: im for im in inst["images"]}
        biggest = {}
        for a in inst["annotations"]:
            name = names[a["category_id"]]
            if name not in CATEGORIES:
                continue
            im = images[a["image_id"]]
            frac = a["area"] / (im["width"] * im["height"])
            if frac > biggest.get(a["image_id"], (0, None))[0]:
                biggest[a["image_id"]] = (frac, name)
        captions = defaultdict(list)
        for c in caps["annotations"]:
            captions[c["image_id"]].append(" ".join(c["caption"].split()))
        for img_id, (frac, name) in biggest.items():
            im = images[img_id]
            lic = cc_by(licenses[im["license"]])
            if frac < MIN_AREA_FRAC or lic is None or not captions[img_id]:
                continue
            image_id = f"coco_{img_id}"
            category, coarse = CATEGORIES[name]
            rows.append({"image_id": image_id, "group_id": image_id, "title": captions[img_id][0].lower(),
                         "captions": captions[img_id], "category": category, "category_coarse": coarse,
                         "license": lic, "license_url": licenses[im["license"]],
                         "attribution": f"Flickr photo {im.get('flickr_url', '')}, via MS COCO (Lin et al., 2014)",
                         "source_url": im["coco_url"], "file": str(WORK / "images" / im["file_name"]),
                         "split": split_of(image_id)})
print(f"{len(rows)} CC BY room photos with a home object >= {MIN_AREA_FRAC:.0%} of the photo")

with ThreadPoolExecutor(16) as pool:
    ok = list(pool.map(lambda r: fetch(r["source_url"], Path(r["file"])), rows))
rows = [r for r, good in zip(rows, ok) if good]
print(f"downloaded {len(rows)} images")

# %% ---------------------------------------------------------------------------

# 4. Save to the data folder: JSONL shards per split (original JPEG bytes, not re-encoded), plus a manifest
files = {r["image_id"]: r["file"] for r in rows}   # local scratch paths, not saved
counts, shards = {}, {}
for split in ("train", "validation"):
    part = [{k: v for k, v in r.items() if k not in ("split", "file")} for r in rows if r["split"] == split]
    counts[split] = {"rows": len(part)}
    shards[split] = []
    for start in range(0, len(part), SHARD_ROWS):
        lines = [json.dumps(r | {"image_b64": base64.b64encode(Path(files[r["image_id"]]).read_bytes()).decode()})
                 for r in part[start:start + SHARD_ROWS]]
        path = f"{run_out}/{split}/part-{len(shards[split]):05d}.jsonl"
        assert fs.put(path, "\n".join(lines) + "\n", overwrite=True), path
        shards[split].append(path)
    print(split, counts[split], f"{len(shards[split])} shards")

manifest = {
    "source": SOURCE, "run_id": RUN_ID, "run_out": run_out,
    "format": "JSONL; one row per photo; image is the original COCO JPEG as base64 in `image_b64`",
    "columns": ["image_id", "group_id", "title", "captions", "category", "category_coarse", "license",
                "license_url", "attribution", "source_url", "image_b64"],
    "counts": counts, "shards": shards,
    "license": "Images: per-row CC BY (Flickr); captions: CC BY 4.0 (COCO Consortium)",
    "attribution": "MS COCO (Lin et al., 2014); per-photo Flickr link in `attribution`",
    "settings": {"min_area_frac": MIN_AREA_FRAC, "val_fraction": VAL_FRACTION, "coco_splits": SPLITS},
}
assert fs.put(f"{run_out}/manifest.json", json.dumps(manifest, indent=2), overwrite=True)
print("saved:", run_out)
print([f.path for f in fs.ls(run_out)])
