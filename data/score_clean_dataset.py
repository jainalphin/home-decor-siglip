# # Score the clean dataset: image-text match + near-duplicates (GPU, ~15 min, no OpenAI)
#
# Two standard cleaning steps (LAION / DataComp / SemDeDup), done with the **base** SigLIP 2 model
# (the model training starts from):
#
# 1. **Match score**: cosine similarity of each training image with its own text
#    (ABO and DF crops: `vlm_caption`; DF scenes: `title`). The lowest pairs are shown so you can see whether
#    they are really mismatched before choosing a cut-off. Cut-offs are per source, because the base model scores
#    studio photos, renders and rooms differently.
# 2. **Near-duplicates**: pairs of training images that look almost the same but count as *different* items in
#    training (two views of the same product or crops of the same 3D model are wanted and are not counted).
#
# Nothing is dropped until the last cell, which uses the cut-offs you set after looking at the pages.
# Output: `decore/checks/score/<run>/` (scores, duplicate pairs, `exclude.txt` for `build_clean_dataset.py`).


import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "finetuning")]
from fsutil import fs, show_html   # file helpers and HTML review pages
# %% ---------------------------------------------------------------------------

# 1. Settings and helpers
import base64, io, json, random
from collections import Counter, defaultdict
from datetime import datetime, timezone

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor

DATA_ROOT = "decore"
CLEAN = f"{DATA_ROOT}/clean/home-decore/20261108T060636Z"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = f"{DATA_ROOT}/checks/score/{RUN_ID}"
MODEL = "google/siglip2-base-patch16-224"
TEXT_FIELD = {"abo": "vlm_caption", "deepfurniture_crops": "vlm_caption", "deepfurniture_scenes": "title"}
IDENTITY = {"abo": "group_id", "deepfurniture_crops": "furniture_id", "deepfurniture_scenes": "scene_id"}
DUP_MIN = 0.90          # duplicate pairs above this are saved; the cut-off itself is chosen in the last cell
SEED = 0
assert torch.cuda.is_available(), "A GPU is required"
device, dtype = "cuda", torch.bfloat16
Image.MAX_IMAGE_PIXELS = None

model = AutoModel.from_pretrained(MODEL).to(device).eval()
processor = AutoProcessor.from_pretrained(MODEL)


def read_text(path):
    return fs.head(path, fs.ls(path)[0].size)


def shard_paths(folder):
    return sorted(f.path for f in fs.ls(folder) if f.name.endswith(".jsonl"))


def put(path, text):
    assert fs.put(path, text, overwrite=True), path


def pooled(value):
    return value if hasattr(value, "shape") else value.pooler_output


def to_pil(b64):
    return ImageOps.exif_transpose(Image.open(io.BytesIO(base64.b64decode(b64)))).convert("RGB")


def thumb_b64(image, side=112):
    image = image.copy(); image.thumbnail((side, side)); buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


@torch.no_grad()
def embed(images, texts):
    pixels = processor(images=images, return_tensors="pt").pixel_values.to(device)
    tokens = processor(text=[t.lower() for t in texts], padding="max_length", truncation=True, max_length=64,
                       return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.autocast("cuda", dtype=dtype):
        im = F.normalize(pooled(model.get_image_features(pixel_values=pixels)).float(), dim=-1)
        tx = F.normalize(pooled(model.get_text_features(**tokens)).float(), dim=-1)
    return im.cpu(), tx.cpu()

print("output:", OUT)

# %% ---------------------------------------------------------------------------

# 2. Embed every clean training image and its text (one pass over the data folder; only small thumbnails are kept)
data = {}
for source, field in TEXT_FIELD.items():
    meta, image_emb, scores = [], [], []
    for path in shard_paths(f"{CLEAN}/{source}/train"):
        rows = [json.loads(l) for l in read_text(path).splitlines() if l]
        for start in range(0, len(rows), 128):
            batch = rows[start:start + 128]
            images = [to_pil(r["image_b64"]) for r in batch]
            texts = [str(r.get(field) or r.get("title") or "") for r in batch]
            im, tx = embed(images, texts)
            image_emb.append(im); scores += (im * tx).sum(-1).tolist()
            for r, image, text in zip(batch, images, texts):
                meta.append({"key": f"{source}/train:{r['row_index']}", "identity": str(r.get(IDENTITY[source])),
                             "text": text, "is_main": bool(r.get("is_main")), "scene_group": r.get("scene_group"),
                             "thumb": thumb_b64(image)})
    data[source] = {"meta": meta, "emb": torch.cat(image_emb), "score": torch.tensor(scores)}
    s = data[source]["score"]
    print(f"{source}: {len(meta):,} rows | match score percentiles 1/5/25/50:",
          [round(s.quantile(q).item(), 3) for q in (.01, .05, .25, .5)], flush=True)

put(f"{OUT}/scores.jsonl", "\n".join(json.dumps({"key": m["key"], "score": round(float(sc), 4)})
                                      for d in data.values() for m, sc in zip(d["meta"], d["score"])) + "\n")

# %% ---------------------------------------------------------------------------

# 3. Look: the lowest-scoring pairs per source, and pairs around the 2nd-5th percentile (where a cut-off would go)


def card(m, note):
    return (f"<div style='display:inline-block;width:150px;vertical-align:top;margin:3px;font-size:10px'>"
            f"<img src='data:image/jpeg;base64,{m['thumb']}' width=112><br><b>{note}</b><br>{m['text'][:110]}</div>")


html = []
for source, d in data.items():
    order = d["score"].argsort().tolist()
    n = len(order)
    html.append(f"<h3>{source}: 40 lowest</h3>" + "".join(card(d["meta"][i], f"{d['score'][i]:.3f}") for i in order[:40]))
    band = random.Random(SEED).sample(order[int(n * .02):int(n * .05)], min(20, n))
    html.append(f"<h4>{source}: 20 random from the 2nd-5th percentile</h4>"
                + "".join(card(d["meta"][i], f"{d['score'][i]:.3f}") for i in band))
show_html("".join(html))

# %% ---------------------------------------------------------------------------

# 4. Near-duplicate pairs: image similarity >= DUP_MIN between rows that training treats as different items
dups = {}
for source, d in data.items():
    emb = d["emb"].to(device, torch.float16)
    ident = [m["identity"] for m in d["meta"]]
    codes = {v: k for k, v in enumerate(sorted(set(ident)))}
    ident_t = torch.tensor([codes[v] for v in ident], device=device)
    pairs = []
    for start in range(0, len(emb), 4096):
        sims = emb[start:start + 4096] @ emb.T
        rows = torch.arange(start, start + len(sims), device=device)[:, None]
        cols = torch.arange(len(emb), device=device)[None, :]
        mask = (cols > rows) & (ident_t[None, :] != ident_t[start:start + len(sims)][:, None]) & (sims >= DUP_MIN)
        for i, j in mask.nonzero().tolist():
            pairs.append((start + i, j, round(float(sims[i, j]), 4)))
        assert len(pairs) < 3_000_000, f"{source}: too many pairs >= {DUP_MIN}; raise DUP_MIN and re-run this cell"
    dups[source] = pairs
    bands = Counter(min(int((s - DUP_MIN) / 0.02), 4) for _, _, s in pairs)
    print(f"{source}: {len(pairs):,} pairs >= {DUP_MIN} | by band (0.90/.92/.94/.96/.98+):",
          [bands[b] for b in range(5)], flush=True)

put(f"{OUT}/dup_pairs.jsonl", "\n".join(json.dumps({"a": data[s]["meta"][i]["key"], "b": data[s]["meta"][j]["key"],
                                                     "sim": sim}) for s, p in dups.items() for i, j, sim in p) + "\n")

html = []
for source, pairs in dups.items():
    for lo, hi in ((0.90, 0.93), (0.93, 0.96), (0.96, 0.98), (0.98, 1.01)):
        band = [p for p in pairs if lo <= p[2] < hi]
        pick = random.Random(SEED).sample(band, min(6, len(band)))
        html.append(f"<h4>{source}: similarity {lo}-{min(hi, 1.0)} ({len(band):,} pairs), 6 random</h4>" + "".join(
            f"<div style='display:inline-block;margin:4px;border:1px solid #ccc;font-size:10px'>"
            f"<img src='data:image/jpeg;base64,{data[source]['meta'][i]['thumb']}' width=112>"
            f"<img src='data:image/jpeg;base64,{data[source]['meta'][j]['thumb']}' width=112><br>{s:.3f}</div>"
            for i, j, s in pick))
show_html("".join(html))

# %% ---------------------------------------------------------------------------

# 5. Decide after looking at the pages above, then run this cell.
# LOW_PCT: drop the lowest X percent per source by match score (0 = keep all).
# DUP_T:   near-duplicate cut-off; in each group of near-duplicates one row is kept (ABO: main photo first).
LOW_PCT = {"abo": 0, "deepfurniture_crops": 0, "deepfurniture_scenes": 0}
DUP_T = {"abo": 1.01, "deepfurniture_crops": 1.01, "deepfurniture_scenes": 1.01}   # 1.01 = no dedup

drop, reasons = set(), Counter()
for source, d in data.items():
    n = len(d["meta"])
    if LOW_PCT[source]:
        for i in d["score"].argsort()[:int(n * LOW_PCT[source] / 100)].tolist():
            drop.add(d["meta"][i]["key"]); reasons[f"{source}: low match"] += 1
    parent = list(range(n))   # union-find over near-duplicate pairs

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j, s in dups[source]:
        if s >= DUP_T[source]:
            parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    for members in groups.values():
        if len(members) > 1:
            members.sort(key=lambda i: (not d["meta"][i]["is_main"], -float(d["score"][i])))
            for i in members[1:]:
                if d["meta"][i]["key"] not in drop:
                    drop.add(d["meta"][i]["key"]); reasons[f"{source}: near duplicate"] += 1
print(dict(reasons), "| total:", len(drop))
put(f"{OUT}/exclude.txt", "\n".join(sorted(drop)) + "\n")
put(f"{OUT}/settings.json", json.dumps({"LOW_PCT": LOW_PCT, "DUP_T": DUP_T, "counts": dict(reasons)}, indent=2))
print("saved:", f"{OUT}/exclude.txt")
