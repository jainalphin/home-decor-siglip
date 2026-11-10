# # Scene leak check: is the same DeepFurniture room in train and test (different camera angle)?
#
# The leak check removed near-identical images (similarity >= 0.97). This script looks for the next level down:
# the same room rendered from another angle. Steps:
# 1. For each test scene used in eval cell 4b, find the most similar TRAIN scene (base SigLIP 2, same judge as the leak check).
# 2. Count test scenes per similarity band and show pairs per band: **you decide by eye where "same room" starts**.
# 3. Re-score the 4b scene tests (caption -> room, query -> room) on all test scenes vs. only test scenes with no
#    close train scene. If the trained models' gain shrinks a lot on the safe subset, the gain was partly leakage.
#
# Train scenes = source train split minus the leak-check drops (what v3-baseline saw; the clean runs saw a subset).
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
import base64, hashlib, io, json, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HOME_DECOR_WORK", "/tmp/decore/eval")
sys.path.insert(0, HERE)   # train_v3.py next to this script, or ...
sys.path.insert(1, os.path.join(HERE, "..", "finetuning"))   # ... in ../finetuning (repo layout)
import train_v3 as t
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModel, AutoProcessor

ROOT = "decore"
SRC = f"{ROOT}/sources/home-decore/20261104T130155Z"
LEAK = f"{ROOT}/checks/leak/20261105T055610Z/home-decore/exclude_v2.txt"
RECAP_TEST = f"{ROOT}/checks/recaption/test_v1"
MODELS_DIR = f"{ROOT}/models"
OUT_DIR = f"{ROOT}/checks/scene_leak/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
MODEL_REPO = "AlphJain/home-decor-siglip"
MODELS = ["base", "v3-baseline", "run:v3-clean-nostyle"]
BANDS = [0.97, 0.95, 0.93, 0.90, 0.85, 0.0]      # lower edges; >= 0.97 should be empty (leak check)
SAFE_BELOW = [0.95, 0.93, 0.90, 0.85]            # re-score on test scenes whose nearest train scene is below this
PAIRS_PER_BAND, BOOTSTRAP, SEED = 8, 1000, 0
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
    print("loaded", name)
    return (AutoModel.from_pretrained(source, dtype=torch.float32).to(device).eval(),
            AutoProcessor.from_pretrained(source))


def embed(model, processor, b64s):
    return t.embed_images(model, processor, [lambda b=b: {"bytes": base64.b64decode(b)} for b in b64s], device, dtype)


def thumb(b64, size=200):
    im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    im.thumbnail((size, size))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=80)
    return f"<img src='data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}'>"

# %% ---------------------------------------------------------------------------

# 2. Load scenes: test scenes exactly as in eval cell 4b, train scenes minus the leak-check drops
leak = set(read_file(LEAK).split())
train = [r for r in read_folder(f"{SRC}/deepfurniture_scenes/train")
         if f"deepfurniture_scenes/train:{r['row_index']}" not in leak]
recap = {r["key"]: r for r in read_folder(f"{RECAP_TEST}/deepfurniture_scenes")}
test = []
for r in read_folder(f"{SRC}/deepfurniture_scenes/test"):
    rec = recap.get(f"deepfurniture_scenes/test:{r['row_index']}")
    if rec and rec.get("rc_caption") and not rec.get("rc_wrong_item"):
        test.append((r, rec))
print(f"train scenes {len(train):,} | test scenes {len(test):,} (4b used 1,290)")

# %% ---------------------------------------------------------------------------

# 3. Nearest train scene for every test scene (base SigLIP 2)
model, processor = load_model("base")
train_emb = embed(model, processor, [r["image_b64"] for r in train])
test_emb = embed(model, processor, [r["image_b64"] for r, _ in test])
del model; torch.cuda.empty_cache()
sim = test_emb @ train_emb.T
nearest_sim, nearest = sim.max(1)
nearest_sim, nearest = nearest_sim.numpy(), nearest.numpy()

edges = BANDS
band_of = lambda s: next(e for e in edges if s >= e)
counts = {f">= {e:.2f}" if e else f"< {edges[-2]:.2f}": int(sum(band_of(s) == e for s in nearest_sim)) for e in edges}
print("test scenes by similarity to their nearest train scene:", json.dumps(counts, indent=1))
for cut in SAFE_BELOW:
    print(f"  nearest train scene >= {cut}: {int((nearest_sim >= cut).sum()):5,} of {len(test):,} "
          f"({(nearest_sim >= cut).mean():.1%})")
assert fs.put(f"{OUT_DIR}/nearest.jsonl", "\n".join(json.dumps(
    {"test_row": r["row_index"], "train_row": train[int(j)]["row_index"], "sim": round(float(s), 4)})
    for (r, _), j, s in zip(test, nearest, nearest_sim)) + "\n", overwrite=True)

# %% ---------------------------------------------------------------------------

# 4. Look: test scene | its nearest train scene, a few pairs per band. Where do pairs stop being the same room?
rng = np.random.default_rng(SEED)
html = []
for hi, lo in zip([1.01] + edges[:-1], edges):
    idx = np.where((nearest_sim >= lo) & (nearest_sim < hi))[0]
    if not len(idx):
        continue
    pick = rng.choice(idx, min(PAIRS_PER_BAND, len(idx)), replace=False)
    html.append(f"<h3>{lo:.2f} - {min(hi, 1):.2f}: {len(idx)} test scenes</h3><table border=1 cellpadding=2>"
                "<tr><th>sim</th><th>test</th><th>nearest train</th></tr>")
    for i in pick:
        html.append(f"<tr><td>{nearest_sim[i]:.3f}</td><td>{thumb(test[i][0]['image_b64'])}</td>"
                    f"<td>{thumb(train[int(nearest[i])]['image_b64'])}</td></tr>")
    html.append("</table>")
show_html("<h2>Same room? (test | nearest train)</h2>" + "".join(html))

# %% ---------------------------------------------------------------------------

# 5. Re-score the scene tests (full gallery of test scenes) on all test scenes vs. the safe subsets
images = [r["image_b64"] for r, _ in test]
tests = {"scenes: caption": [(rec["rc_caption"], i) for i, (_, rec) in enumerate(test)],
         "scenes: query": [(q, i) for i, (_, rec) in enumerate(test) for q in rec.get("rc_queries", [])]}
ranks = {name: {} for name in tests}
for model_name in MODELS:
    model, processor = load_model(model_name)
    ge = test_emb if model_name == "base" else embed(model, processor, images)
    for name, pairs in tests.items():
        qe = t.embed_texts(model, processor, [text for text, _ in pairs], device, dtype)
        target = torch.tensor([i for _, i in pairs])
        s = qe @ ge.T
        ranks[name][model_name] = ((s > s[torch.arange(len(pairs)), target][:, None]).sum(1) + 1).numpy()
    del model; torch.cuda.empty_cache()


def diff(a, b, k):
    ha, hb = (a <= k).astype(float), (b <= k).astype(float)
    idx = np.random.default_rng(SEED).integers(0, len(ha), size=(BOOTSTRAP, len(ha)))
    d = ha[idx].mean(1) - hb[idx].mean(1)
    return {"diff": round(float(ha.mean() - hb.mean()), 4),
            "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]}


results = {}
for name, pairs in tests.items():
    scene_sim = np.array([nearest_sim[i] for _, i in pairs])
    for label, keep in [("all", np.ones(len(pairs), bool))] + [(f"nearest < {c}", scene_sim < c) for c in SAFE_BELOW]:
        r = {m: ranks[name][m][keep] for m in MODELS}
        results[f"{name} | {label}"] = {
            "queries": int(keep.sum()),
            **{m: {"R@1": round(float((v <= 1).mean()), 4), "R@10": round(float((v <= 10).mean()), 4)}
               for m, v in r.items()},
            "nostyle - base R@10": diff(r[MODELS[2]], r["base"], 10),
            "nostyle - v3-baseline R@10": diff(r[MODELS[2]], r[MODELS[1]], 10),
        }
for key, value in results.items():
    print(f"{key:40s} n={value['queries']:5,} | " + " | ".join(
        f"{m} {value[m]['R@1']:.3f}/{value[m]['R@10']:.3f}" for m in MODELS)
        + f" | nostyle-base R@10 {value['nostyle - base R@10']['diff']:+.3f} {value['nostyle - base R@10']['ci95']}")
assert fs.put(f"{OUT_DIR}/results.json", json.dumps({"counts": counts, "results": results}, indent=2),
                      overwrite=True)
print("saved:", OUT_DIR)
