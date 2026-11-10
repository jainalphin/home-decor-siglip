# Leak check helpers. Plain Python file (not a script); the `run_leak_check` script loads it with
# `from leak_check import *`, so keep it in the same workspace folder.
# Every training image is compared against every image we evaluate on:
#   - STL-Home: full scenes, the query crops, and the gallery products
#   - AlphJain/home-decore: validation + test splits of all four configs
# A training image is dropped if it is the same picture (pixel sha256), a near-duplicate (dhash), or almost the
# same picture by SigLIP embedding (catches crops, recompression, small edits). A lower "review" threshold only
# reports possible same-product matches (different photo of the same item) for a human look.
# Writes nothing to the data folder: it returns report text, and the calling script saves it with fsutil.fs.put.
# HF_TOKEN (read access) is needed only here, to load the private eval sets.
import base64, hashlib, io, json, os, tarfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps

# ---- settings ----
DATA_REPO = "AlphJain/home-decore"
STL_REPO = "AlphJain/stl-home-eval"
CONFIGS = ["abo", "deepfurniture_crops", "deepfurniture_products", "deepfurniture_scenes"]
MODEL = "google/siglip2-base-patch16-224"   # base model, not a fine-tune, so the check doesn't depend on training

def writable(path):
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return os.access(path, os.W_OK)
    except OSError:
        return False


# Local scratch: $DECORE_WORK if set, else /local_disk0 (not writable on serverless), else /tmp.
WORK = Path(os.environ.get("DECORE_WORK") or ("/local_disk0/decore" if writable("/local_disk0/decore") else "/tmp/decore"))
STL_ROOT = WORK / "stl-home-eval"
DROP_SIM = 0.98       # cosine >= this -> same picture (drop). Different catalog products reached 0.95-0.964
REVIEW_SIM = 0.88     # cosine >= this -> report as possible same product (not dropped)
DHASH_BITS = 6        # hamming distance <= this -> near-duplicate candidate
DHASH_CONFIRM_SIM = 0.97   # ...dropped only if SigLIP also agrees (identical DF renders 0.974+, different doors 0.964)
LOAD_SIDE = 448
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Image.MAX_IMAGE_PIXELS = None
_model = _processor = None


def token():
    value = os.environ.get("HF_TOKEN")
    if not value:
        raise RuntimeError("Set os.environ['HF_TOKEN'] (read access to AlphJain/home-decore and AlphJain/stl-home-eval)")
    return value


def to_pil(value):
    if isinstance(value, Image.Image):
        image = value
    elif isinstance(value, (bytes, bytearray)):
        image = Image.open(io.BytesIO(value))
    elif isinstance(value, dict) and value.get("bytes") is not None:
        image = Image.open(io.BytesIO(value["bytes"]))
    else:
        image = Image.open(value["path"] if isinstance(value, dict) else value)
    image.draft("RGB", (LOAD_SIDE, LOAD_SIDE))
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA", "P"):
        rgba = image.convert("RGBA")
        image = Image.new("RGB", rgba.size, (255, 255, 255))
        image.paste(rgba, mask=rgba.split()[-1])
    image = image.convert("RGB")
    if max(image.size) > LOAD_SIDE:
        image.thumbnail((LOAD_SIDE, LOAD_SIDE))
    return image


def sha256(image):
    return hashlib.sha256(image.tobytes()).hexdigest()   # pixel hash: the same picture saved twice still matches


def dhash(image):
    small = np.asarray(image.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def embed(images, batch_size=64):
    global _model, _processor
    from transformers import AutoModel, AutoProcessor
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    if _model is None:
        _processor = AutoProcessor.from_pretrained(MODEL)
        _model = AutoModel.from_pretrained(MODEL, dtype=dtype).to(DEVICE).eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            pixels = _processor(images=images[start:start + batch_size], return_tensors="pt").pixel_values
            feats = _model.get_image_features(pixel_values=pixels.to(DEVICE, dtype))
            feats = feats if hasattr(feats, "shape") else feats.pooler_output
            out.append(F.normalize(feats.float(), dim=-1).cpu())
    return torch.cat(out)


def fingerprint(loaders, names, batch=512):
    # loaders: functions that return a PIL image. Returns names, sha256, dhash, embeddings.
    shas, hashes, embs = [], [], []
    for start in range(0, len(loaders), batch):
        images = [load() for load in loaders[start:start + batch]]
        shas += [sha256(x) for x in images]
        hashes += [dhash(x) for x in images]
        embs.append(embed(images))
        print(f"  fingerprinted {min(start + batch, len(loaders))}/{len(loaders)}", flush=True)
    return {"names": list(names), "sha": shas, "dhash": np.array(hashes, dtype=np.uint64), "emb": torch.cat(embs)}


def stl_loaders():
    from huggingface_hub import snapshot_download
    if not (STL_ROOT / "queries.jsonl").exists():
        snapshot_download(STL_REPO, repo_type="dataset", token=token(), local_dir=STL_ROOT,
                          allow_patterns=["queries.jsonl", "gallery.jsonl", "images.tar"])
    if not (STL_ROOT / "scenes").exists():
        with tarfile.open(STL_ROOT / "images.tar") as tar:
            tar.extractall(STL_ROOT, filter="data")
    queries = [json.loads(x) for x in (STL_ROOT / "queries.jsonl").read_text().splitlines() if x]
    gallery = [json.loads(x) for x in (STL_ROOT / "gallery.jsonl").read_text().splitlines() if x]

    def crop(q):   # same crop as the STL eval in the trainer (bbox + 10% padding)
        def load():
            image = Image.open(STL_ROOT / "scenes" / f"{q['scene']}.jpg")
            image = ImageOps.exif_transpose(image).convert("RGB")
            w, h = image.size
            l, t, r, b = q["bbox"]
            px, py = (r - l) * .1, (b - t) * .1
            return to_pil(image.crop((max(0, (l - px) * w), max(0, (t - py) * h),
                                      min(w, (r + px) * w), min(h, (b + py) * h))))
        return load

    items = []
    for scene in sorted({q["scene"] for q in queries}):
        items.append((f"stl_scene:{scene}", lambda s=scene: to_pil(STL_ROOT / "scenes" / f"{s}.jpg")))
    for n, q in enumerate(queries):
        items.append((f"stl_query:{n}:{q['scene']}", crop(q)))
    for g in gallery:
        items.append((f"stl_product:{g['product']}", lambda p=g["product"]: to_pil(STL_ROOT / "products" / f"{p}.jpg")))
    return items


def home_decore_loaders():
    from datasets import Image as ImageFeature, load_dataset
    items = []
    for config in CONFIGS:
        for split in ("validation", "test"):
            ds = load_dataset(DATA_REPO, config, split=split, token=token(), cache_dir=str(WORK / "hf_cache"))
            column = next(c for c in ("image", "crop", "scene", "render") if c in ds.column_names)
            ds = ds.cast_column(column, ImageFeature(decode=False))
            for n in range(len(ds)):
                items.append((f"{config}/{split}:{n}", lambda v=ds[n][column]: to_pil(v)))
    return items


def eval_index():
    path = WORK / "eval_index.pt"   # local scratch cache; rebuilt (a few minutes on GPU) if the cluster restarts
    if path.exists():
        return torch.load(path, weights_only=False)
    WORK.mkdir(parents=True, exist_ok=True)
    items = stl_loaders() + home_decore_loaders()
    print(f"building eval index over {len(items)} images")
    index = fingerprint([x[1] for x in items], [x[0] for x in items])
    torch.save(index, path)
    return index


def hamming_min(train_hashes, index_hashes, chunk=256):
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    best, where = [], []
    for start in range(0, len(train_hashes), chunk):
        x = train_hashes[start:start + chunk, None] ^ index_hashes[None, :]
        bits = table[x.view(np.uint8)].reshape(*x.shape, 8).sum(-1)
        best.append(bits.min(1)); where.append(bits.argmin(1))
    return np.concatenate(best), np.concatenate(where)


def merge(parts):
    # Join fingerprints computed shard by shard into one.
    return {"names": [n for p in parts for n in p["names"]], "sha": [x for p in parts for x in p["sha"]],
            "dhash": np.concatenate([p["dhash"] for p in parts]), "emb": torch.cat([p["emb"] for p in parts])}


def check(train, source, load_train):
    """Compare fingerprinted training images to the eval index.
    train: output of fingerprint()/merge(); load_train(name) returns that training image (for the contact sheet).
    Returns keep mask, summary dict, per-row JSONL text, and an HTML contact sheet of the closest pairs."""
    index = eval_index()
    names = train["names"]
    sims, nearest = [], []
    index_emb = index["emb"].to(DEVICE)
    for start in range(0, len(train["emb"]), 4096):
        s = train["emb"][start:start + 4096].to(DEVICE) @ index_emb.T
        value, arg = s.max(1)
        sims.append(value.cpu()); nearest.append(arg.cpu())
    sims, nearest = torch.cat(sims).numpy(), torch.cat(nearest).numpy()
    bits, bits_at = hamming_min(train["dhash"], index["dhash"])
    index_sha = {s: n for n, s in enumerate(index["sha"])}

    rows, keep = [], []
    for n, name in enumerate(names):
        reason = ""
        # Similarity of this image to the one its dhash points at: a real resized/recompressed copy is ~0.95+,
        # while plain low-detail images can share a dhash and still be different pictures (COCO run: 17 such
        # matches, all at 0.57-0.78).
        dhash_sim = float(train["emb"][n] @ index["emb"][int(bits_at[n])])
        if train["sha"][n] in index_sha:
            reason, match = "same_pixels", index_sha[train["sha"][n]]
        elif bits[n] <= DHASH_BITS and dhash_sim >= DHASH_CONFIRM_SIM:
            reason, match = "dhash", int(bits_at[n])
        elif sims[n] >= DROP_SIM:
            reason, match = "embedding", int(nearest[n])
        else:
            match = int(nearest[n])
        keep.append(not reason)
        rows.append({"name": name, "dropped": reason, "max_sim": round(float(sims[n]), 4),
                     "dhash_bits": int(bits[n]), "dhash_pair_sim": round(dhash_sim, 4),
                     "nearest_eval": index["names"][match]})

    by_target = {}
    for r in rows:
        if r["dropped"]:
            kind = r["nearest_eval"].split(":")[0].split("/")[0]
            by_target[kind] = by_target.get(kind, 0) + 1
    summary = {"source": source, "checked": len(rows), "dropped": len(rows) - sum(keep),
               "dropped_by_eval_set": by_target,
               "review_possible_same_product": sum(1 for r in rows if not r["dropped"] and r["max_sim"] >= REVIEW_SIM),
               "thresholds": {"drop_sim": DROP_SIM, "review_sim": REVIEW_SIM, "dhash_bits": DHASH_BITS,
                              "dhash_confirm_sim": DHASH_CONFIRM_SIM},
               "max_sim_percentiles": {str(p): round(float(np.percentile(sims, p)), 4) for p in (50, 90, 99, 99.9)}}
    print(json.dumps(summary, indent=2))
    rows_text = "\n".join(json.dumps(r) for r in rows)
    return keep, summary, rows_text, contact_sheet(load_train, rows, source)


def jpeg_b64(image, side=200):
    image = image.copy()
    image.thumbnail((side, side))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode()


def contact_sheet(load_train, rows, source, count=40, max_dropped=100):
    # Training image left, matched eval image right, as one self-contained HTML page (saved to the data folder as text,
    # opened in a browser or with fsutil.show_html). Every dropped row comes first (up to max_dropped), whatever
    # test dropped it, then the closest kept rows, so hash drops are visible too, not only high-similarity ones.
    by_sim = sorted(range(len(rows)), key=lambda n: -rows[n]["max_sim"])
    order = [n for n in by_sim if rows[n]["dropped"]][:max_dropped] + [n for n in by_sim if not rows[n]["dropped"]][:count]
    eval_items = dict(stl_loaders() + home_decore_loaders()) if order else {}
    cells = []
    for n in order:
        r = rows[n]
        left, right = jpeg_b64(load_train(r["name"])), jpeg_b64(eval_items[r["nearest_eval"]]())
        cells.append(f"<tr><td><img src='data:image/jpeg;base64,{left}'></td>"
                     f"<td><img src='data:image/jpeg;base64,{right}'></td>"
                     f"<td>sim {r['max_sim']:.3f} | dhash bits {r['dhash_bits']} (pair sim {r['dhash_pair_sim']:.3f})<br>{r['dropped'] or 'kept'}<br>"
                     f"{r['name']}<br>{r['nearest_eval']}</td></tr>")
    return (f"<html><body><h3>{source}: dropped pairs first, then the closest kept pairs</h3>"
            f"<table border=1 cellpadding=4>{''.join(cells)}</table></body></html>")

