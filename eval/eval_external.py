# # External test: real photos the compared models never trained on (evaluation only)
#
# | test | query | gallery | correct answer |
# |---|---|---|---|
# | **COCO home** caption -> photo | each human-written COCO caption | all kept COCO home photos | its own photo |
# | **Pix3D** photo -> photo | each real furniture photo | all other Pix3D photos | any photo of the same furniture model |
#
# Data: the existing stored copies (`sources/coco_scenes`, `sources/pix3d`, train + validation splits). base,
# v3-baseline and v3-clean-nostyle never saw them (only the old `v3-real_photos` run trained on them, and it
# is not compared here). COCO keeps indoor home photos only, with the same caption word rule as
# `openai_label_jobs` Part 0. Nothing here is used for training or for picking checkpoints.
#
# Put next to `train_v3.py`. GPU, HF read token (base model + v3-baseline). About 20-30 min.


import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "finetuning")]
from fsutil import fs, show_html   # file helpers and HTML review pages
# %% ---------------------------------------------------------------------------

import os
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN") or ""   # e.g. "hf_..."
assert os.environ["HF_TOKEN"], "Set the HF_TOKEN environment variable"

# %% ---------------------------------------------------------------------------

# 1. Settings and helpers
import base64, hashlib, json, re, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HOME_DECOR_WORK", "/tmp/decore/eval")
sys.path.insert(0, HERE)   # train_v3.py next to this script, or ...
sys.path.insert(1, os.path.join(HERE, "..", "finetuning"))   # ... in ../finetuning (repo layout)
import train_v3 as t
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModel, AutoProcessor

ROOT = "decore"
COCO = f"{ROOT}/sources/coco_scenes/20261104T151655Z"
PIX3D = f"{ROOT}/sources/pix3d/20261104T151426Z"
MODELS_DIR = f"{ROOT}/models"
OUT_DIR = f"{ROOT}/evals/external/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
MODEL_REPO = "AlphJain/home-decor-siglip"
MODELS = ["base", "run:v3-clean-nostyle", "run:v3-clean-oi"]   # first = base; differences are vs base and vs MODELS[1]
BOOTSTRAP, SEED = 1000, 0
token = os.environ["HF_TOKEN"]
t.TOKEN = token
assert torch.cuda.is_available(), "A GPU is required"
device, dtype = torch.device("cuda"), torch.bfloat16
print("results:", OUT_DIR)


def read_file(path):
    return fs.head(path, fs.ls(path)[0].size)


def read_folder(folder):
    return [json.loads(l) for p in sorted(f.path for f in fs.ls(folder) if f.name.endswith(".jsonl"))
            for l in read_file(p).splitlines() if l]


def stored_model_dir(run):
    root = f"{MODELS_DIR}/{run}"
    step = json.loads(read_file(f"{root}/best.json"))["step"]
    rel = f"{root}/checkpoints/step-{int(step):08d}/model"
    local = Path(f"/tmp/decore/eval_models/{run}/step-{int(step):08d}")
    for name in json.loads(read_file(f"{rel}/files.json")):
        if (local / name).exists():
            continue
        manifest = json.loads(read_file(f"{rel}/{name}/manifest.json"))
        data = b"".join(base64.b64decode(read_file(f"{rel}/{name}/{p}")) for p in manifest["parts"])
        assert hashlib.sha256(data).hexdigest() == manifest["sha256"], f"checksum mismatch {name}"
        (local / name).parent.mkdir(parents=True, exist_ok=True)
        (local / name).write_bytes(data)
    return str(local)


def load_model(name):
    if name == "base":
        source = t.BASE_MODEL
    elif name.startswith("run:"):
        source = stored_model_dir(name[4:])
    else:
        step = json.loads(Path(hf_hub_download(MODEL_REPO, f"runs/{name}/best.json", token=token,
                                               force_download=True)).read_text())["step"]
        prefix = f"runs/{name}/checkpoints/step-{step:08d}/model"
        source = str(Path(snapshot_download(MODEL_REPO, token=token, allow_patterns=[f"{prefix}/*"])) / prefix)
    print("loaded", name, flush=True)
    return (AutoModel.from_pretrained(source, dtype=torch.float32).to(device).eval(),
            AutoProcessor.from_pretrained(source))


def embed(model, processor, b64s):
    return t.embed_images(model, processor, [lambda b=b: {"bytes": base64.b64decode(b)} for b in b64s], device, dtype)

# %% ---------------------------------------------------------------------------

# 2. Load the two test sets
# COCO: same indoor-home caption rule as openai_label_jobs Part 0
ROOM_WORDS = ["living room", "bedroom", "kitchen", "dining room", "bathroom", "office", "room", "couch", "sofa",
              "bed", "fireplace", "bookshelf", "bookcase", "armchair", "coffee table", "dresser", "nightstand",
              "lamp", "rug", "curtain", "interior", "hotel", "apartment", "den", "loft"]
OUTDOOR_WORDS = ["street", "sidewalk", "park", "outdoor", "outside", "road", "beach", "field", "grass", "yard",
                 "patio", "parking", "bus", "train", "airport", "snow", "lawn", "garden", "playground", "boat",
                 "ship", "truck", "car", "bike", "bicycle", "skateboard", "horse", "cow", "dog", "cat", "elephant"]
room_re = re.compile(r"\b(" + "|".join(map(re.escape, ROOM_WORDS)) + r")s?\b")
out_re = re.compile(r"\b(" + "|".join(map(re.escape, OUTDOOR_WORDS)) + r")s?\b")
coco = []
for split in ("train", "validation"):
    for r in read_folder(f"{COCO}/{split}"):
        text = " ".join(r.get("captions") or [r.get("title", "")]).lower()
        if r.get("captions") and room_re.search(text) and not out_re.search(text):
            coco.append(r)
coco_pairs = [(c, i) for i, r in enumerate(coco) for c in r["captions"]]
print(f"COCO home photos {len(coco):,} | caption queries {len(coco_pairs):,}")

pix3d = [r for split in ("train", "validation") for r in read_folder(f"{PIX3D}/{split}")]
group_of = np.array([r["group_id"] for r in pix3d])
sizes = Counter(group_of)
pix_queries = np.array([i for i, g in enumerate(group_of) if sizes[g] > 1])   # needs another photo of its model
print(f"Pix3D photos {len(pix3d):,} | models {len(sizes):,} | queries (model has >= 2 photos) {len(pix_queries):,}"
      f" | photos per model median {int(np.median(list(sizes.values())))}, max {max(sizes.values())}")
print("Pix3D categories:", dict(Counter(r["category"] for r in pix3d)))

# %% ---------------------------------------------------------------------------

# 3. Rank with each model
ranks = {"coco: caption -> photo": {}, "pix3d: photo -> same model": {}}
pix_categories = np.array([pix3d[i]["category"] for i in pix_queries])
for name in MODELS:
    model, processor = load_model(name)
    # COCO: text -> image, one correct photo per caption
    ge = embed(model, processor, [r["image_b64"] for r in coco])
    qe = t.embed_texts(model, processor, [c for c, _ in coco_pairs], device, dtype)
    target = torch.tensor([i for _, i in coco_pairs])
    s = qe @ ge.T
    ranks["coco: caption -> photo"][name] = ((s > s[torch.arange(len(target)), target][:, None]).sum(1) + 1).numpy()
    # Pix3D: image -> image, correct = any other photo of the same furniture model (the query itself excluded)
    pe = embed(model, processor, [r["image_b64"] for r in pix3d])
    out = []
    same = torch.tensor(group_of[:, None] == group_of[None, :])
    for start in range(0, len(pix_queries), 512):
        q = torch.tensor(pix_queries[start:start + 512])
        s = pe[q] @ pe.T
        s[torch.arange(len(q)), q] = -2                            # never match itself
        rel = same[q].clone(); rel[torch.arange(len(q)), q] = False
        best = torch.where(rel, s, torch.full_like(s, -3)).max(1).values
        out.append(((s > best[:, None]).sum(1) + 1).numpy())
    ranks["pix3d: photo -> same model"][name] = np.concatenate(out)
    del model; torch.cuda.empty_cache()
    print(name, {test: {k: round(float((r[name] <= k).mean()), 3) for k in (1, 10)} for test, r in ranks.items()},
          flush=True)

# %% ---------------------------------------------------------------------------

# 4. Results with 95% bootstrap intervals for the differences


def diff(a, b, k):
    ha, hb = (a <= k).astype(float), (b <= k).astype(float)
    idx = np.random.default_rng(SEED).integers(0, len(ha), size=(BOOTSTRAP, len(ha)))
    d = ha[idx].mean(1) - hb[idx].mean(1)
    return {"diff": round(float(ha.mean() - hb.mean()), 4),
            "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]}


results = {}
for test, r in ranks.items():
    results[test] = {"queries": len(r[MODELS[0]]),
                     **{m: {"R@1": round(float((v <= 1).mean()), 4), "R@10": round(float((v <= 10).mean()), 4),
                            "MRR": round(float((1 / v).mean()), 4)} for m, v in r.items()},
                     **{f"{m} - base R@{k}": diff(r[m], r["base"], k) for m in MODELS[1:] for k in (1, 10)},
                     **{f"{m} - {MODELS[1]} R@{k}": diff(r[m], r[MODELS[1]], k) for m in MODELS[2:] for k in (1, 10)}}
for test, res in results.items():
    print(f"\n{test}  (queries {res['queries']:,})")
    for m in MODELS:
        print(f"  {m:22s} R@1 {res[m]['R@1']:.3f}  R@10 {res[m]['R@10']:.3f}  MRR {res[m]['MRR']:.3f}")
    for key, value in res.items():
        if " - " in key:
            print(f"  {key:40s} {value['diff']:+.3f}  {value['ci95']}")

print("\nPix3D per category, R@1 / R@10:")
print(f"{'category':12s} {'n':>5s}  " + "  ".join(f"{m[-16:]:>16s}" for m in MODELS))
per_category = {}
for cat in sorted(set(pix_categories)):
    keep = pix_categories == cat
    per_category[cat] = {m: [round(float((v[keep] <= k).mean()), 3) for k in (1, 10)]
                         for m, v in ranks["pix3d: photo -> same model"].items()}
    print(f"{cat:12s} {int(keep.sum()):5d}  " + "  ".join(f"{per_category[cat][m][0]:.3f} / {per_category[cat][m][1]:.3f}"
                                                         for m in MODELS))
assert fs.put(f"{OUT_DIR}/results.json", json.dumps({"results": results, "pix3d_per_category": per_category},
                                                           indent=2), overwrite=True)
print("saved:", OUT_DIR)
