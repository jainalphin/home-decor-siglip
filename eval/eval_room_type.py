# # External test 2: room type on real photos (Places365 + MIT Indoor67), evaluation only
#
# Short decor-style phrases ("a bedroom") on **real** room photos. Together with the COCO test it separates
# two explanations for the COCO drop: real photos (then this test drops too) or long human sentences (then it doesn't).
#
# | dataset | images used | labels | license |
# |---|---|---|---|
# | Places365 validation (256 px) | home room types only (100 photos each) | room type | research use, images owned by their authors |
# | MIT Indoor67 | home room types only | room type (folder name) | "for research purposes only" |
#
# Test: zero-shot room classification among the home room types (top-1 / top-3 accuracy, per room type) for
# base, v3-baseline and v3-clean-nostyle, with 95% bootstrap intervals for the differences.
# Downloads to local disk only (Places365 val 501 MB via torchvision, Indoor67 2.4 GB); nothing is stored in the data folder
# except the result numbers. Never used for training. Put next to `train_v3.py`. GPU, ~30-40 min.


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
import base64, hashlib, json, sys, tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
import torch
import torchvision

os.environ.setdefault("HOME_DECOR_WORK", "/tmp/decore/eval")
sys.path.insert(0, HERE)   # train_v3.py next to this script, or ...
sys.path.insert(1, os.path.join(HERE, "..", "finetuning"))   # ... in ../finetuning (repo layout)
import train_v3 as t
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModel, AutoProcessor

ROOT = "decore"
MODELS_DIR = f"{ROOT}/models"
OUT_DIR = f"{ROOT}/evals/room_type/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
MODEL_REPO = "AlphJain/home-decor-siglip"
MODELS = ["base", "run:v3-clean-nostyle", "run:v3-clean-oi"]
LOCAL = Path("/local_disk0/tmp/room_type") if Path("/local_disk0").exists() else Path("/tmp/room_type")
LOCAL.mkdir(parents=True, exist_ok=True)
USE_INDOOR67 = False   # MIT's official tar URL returned 404 (2026-11-09); Places365 alone covers the home room types
INDOOR67_TAR = "http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar"
# Home room types. Names not found in a dataset are skipped and printed (no guessing).
# value = the phrase used in the prompt
PLACES_ROOMS = {"bedroom": "bedroom", "living_room": "living room", "kitchen": "kitchen", "dining_room": "dining room",
                "bathroom": "bathroom", "home_office": "home office", "nursery": "nursery", "childs_room": "kids room",
                "closet": "walk-in closet", "dorm_room": "dorm room", "attic": "attic", "basement": "basement",
                "corridor": "hallway", "pantry": "pantry", "utility_room": "utility room",
                "television_room": "tv room", "playroom": "playroom", "home_theater": "home theater",
                "staircase": "staircase", "entrance_hall": "entrance hall", "wet_bar": "home bar",
                "garage/indoor": "garage"}
INDOOR67_ROOMS = {"bedroom": "bedroom", "livingroom": "living room", "kitchen": "kitchen", "dining_room": "dining room",
                  "bathroom": "bathroom", "children_room": "kids room", "closet": "walk-in closet",
                  "corridor": "hallway", "pantry": "pantry", "garage": "garage", "stairscase": "staircase",
                  "winecellar": "wine cellar", "gameroom": "game room", "nursery": "nursery", "lobby": "entrance hall"}
PROMPTS = ["a photo of a {}.", "a {} interior", "{}"]   # averaged per room type (standard zero-shot ensembling)
BOOTSTRAP, SEED = 1000, 0
token = os.environ["HF_TOKEN"]
t.TOKEN = token
assert torch.cuda.is_available(), "A GPU is required"
device, dtype = torch.device("cuda"), torch.bfloat16
print("results:", OUT_DIR, "| local:", LOCAL)


def read_file(path):
    return fs.head(path, fs.ls(path)[0].size)


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

# %% ---------------------------------------------------------------------------

# 2. Download and select home room types
# Places365 validation via torchvision (official URLs + checksums): 36,500 photos, 100 per category
places = torchvision.datasets.Places365(str(LOCAL / "places365"), split="val", small=True, download=True)
names = {c.split("/", 2)[2]: i for i, c in enumerate(places.classes)}   # "/b/bedroom" -> "bedroom"
missing = [r for r in PLACES_ROOMS if r not in names]
rooms_p = {r: p for r, p in PLACES_ROOMS.items() if r in names}
print("Places365 room types used:", len(rooms_p), "| not in Places365 (skipped):", missing)
wanted = {names[r]: r for r in rooms_p}
places_items = [(path, wanted[idx]) for path, idx in places.imgs if idx in wanted]

tests = {"places365": (places_items, rooms_p)}

# MIT Indoor67: one folder per category (off by default, see USE_INDOOR67)
if USE_INDOOR67:
    tar_path = LOCAL / "indoorCVPR_09.tar"
    if not tar_path.exists():
        with requests.get(INDOOR67_TAR, stream=True, timeout=120) as response:
            response.raise_for_status()
            with open(tar_path.with_suffix(".part"), "wb") as out:
                for chunk in response.iter_content(1 << 22):
                    out.write(chunk)
        tar_path.with_suffix(".part").rename(tar_path)
    extract = LOCAL / "indoor67"
    if not extract.exists():
        with tarfile.open(tar_path) as tar:
            tar.extractall(extract, filter="data")
    folders = {p.name: p for p in extract.rglob("*") if p.is_dir() and any(p.glob("*.jpg"))}
    missing = [r for r in INDOOR67_ROOMS if r not in folders]
    rooms_i = {r: p for r, p in INDOOR67_ROOMS.items() if r in folders}
    print("Indoor67 room types used:", len(rooms_i), "| not in Indoor67 (skipped):", missing)
    indoor_items = [(str(f), r) for r in rooms_i for f in sorted(folders[r].glob("*.jpg"))]
    tests["indoor67"] = (indoor_items, rooms_i)

for name, (items, rooms) in tests.items():
    print(f"{name}: {len(items):,} photos | per room type: {dict(Counter(r for _, r in items))}")

# %% ---------------------------------------------------------------------------

# 3. Zero-shot room classification with each model
from PIL import Image


def open_rgb(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:   # a few Indoor67 files may be unreadable: they are skipped and counted below
        return None


broken = {}
for name, (items, rooms) in tests.items():
    ok = [(p, r) for p, r in items if open_rgb(p) is not None]
    broken[name] = len(items) - len(ok)
    tests[name] = (ok, rooms)
print("unreadable photos skipped:", broken)

correct = {name: {} for name in tests}   # name -> model -> (top1 bool array, top3 bool array, labels)
for model_name in MODELS:
    model, processor = load_model(model_name)
    for name, (items, rooms) in tests.items():
        room_list = list(rooms)
        text = t.embed_texts(model, processor, [p.format(rooms[r]) for r in room_list for p in PROMPTS], device, dtype)
        class_emb = torch.nn.functional.normalize(text.view(len(room_list), len(PROMPTS), -1).mean(1), dim=-1)
        image_emb = t.embed_images(model, processor, [lambda p=p: open_rgb(p) for p, _ in items], device, dtype)
        scores = image_emb @ class_emb.T
        labels = torch.tensor([room_list.index(r) for _, r in items])
        top3 = scores.topk(3, dim=1).indices
        correct[name][model_name] = ((top3[:, 0] == labels).numpy(), (top3 == labels[:, None]).any(1).numpy(),
                                     labels.numpy())
    del model; torch.cuda.empty_cache()
    print(model_name, {n: round(float(c[model_name][0].mean()), 3) for n, c in correct.items()}, flush=True)

# %% ---------------------------------------------------------------------------

# 4. Results


def diff(a, b):
    idx = np.random.default_rng(SEED).integers(0, len(a), size=(BOOTSTRAP, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return {"diff": round(float(a.mean() - b.mean()), 4),
            "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]}


results = {}
for name, by_model in correct.items():
    rooms = list(tests[name][1])
    res = {"photos": len(tests[name][0]), "room_types": len(rooms),
           **{m: {"top1": round(float(v[0].mean()), 4), "top3": round(float(v[1].mean()), 4)} for m, v in by_model.items()},
           **{f"{m} - base top1": diff(by_model[m][0].astype(float), by_model["base"][0].astype(float)) for m in MODELS[1:]},
           f"{MODELS[2]} - {MODELS[1]} top1": diff(by_model[MODELS[2]][0].astype(float), by_model[MODELS[1]][0].astype(float)),
           "per_room_top1": {r: {m: round(float(v[0][v[2] == k].mean()), 3) for m, v in by_model.items()}
                             for k, r in enumerate(rooms)}}
    results[name] = res
    print(f"\n{name}: {res['photos']:,} photos, {res['room_types']} room types")
    for m in MODELS:
        print(f"  {m:22s} top-1 {res[m]['top1']:.3f}  top-3 {res[m]['top3']:.3f}")
    for key, value in res.items():
        if " - " in key:
            print(f"  {key:42s} {value['diff']:+.3f}  {value['ci95']}")
    print(f"  {'room type':16s} " + "  ".join(f"{m[-12:]:>12s}" for m in MODELS))
    for r, row in res["per_room_top1"].items():
        print(f"  {r:16s} " + "  ".join(f"{row[m]:12.3f}" for m in MODELS))
assert fs.put(f"{OUT_DIR}/results.json", json.dumps({"results": results, "unreadable": broken,
                                                           "prompts": PROMPTS}, indent=2), overwrite=True)
print("saved:", OUT_DIR)
