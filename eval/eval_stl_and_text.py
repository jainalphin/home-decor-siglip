# # Final test: base vs. trained models on the held-out STL queries, plus a gallery of mistakes
#
# - **Fair test:** the 2,250 STL queries that were **not** used to pick checkpoints (the 500 used are listed in
#   each run's `stl_selection_queries.json`), against the full 19,253-product gallery.
# - **Models:** base SigLIP 2, `v3-baseline` best, `v3-clean` best (loaded from the HF model repo).
# - **Natural-language search (cell 4b):** held-out TEST images searched with the new-style captions and
#   queries (needs `data/recaption.py` run with `SPLIT = "test"` first). Validation is used to pick checkpoints.
# - **Reports:** R@1 / R@10 / R@50 / MRR overall and per category, and 95% bootstrap intervals for the
#   differences (is a gap real or noise?).
# - **Mistakes:** 60 random held-out queries the best model gets wrong at rank 1: query crop, the correct
#   product, and the model's top 3. Use it to sort misses into look-alike / wrong type / bad crop.
# - **Blind checks (no STL labels):** image search (cells 6-7) and text search with 60 shopping phrases
#   (cells 8-9), base vs trained, judged by eye.
# - **Training-data check (cells 10-12):** rows with image + all text fields, sample training pairs, and noise
#   counts for ABO keywords / seller-vs-VLM material and colour.
#
# Put this script next to `train_v3.py` (it reuses the trainer's exact crop and embedding code).
# Needs a **GPU** and an HF read token. Results go to the data folder under `decore/evals/`.


import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "finetuning")]
from fsutil import fs, show_html   # file helpers and HTML review pages
# %% ---------------------------------------------------------------------------

# HF token: read access to AlphJain/home-decor-siglip and AlphJain/stl-home-eval.
# Set it as an environment variable before running; never write it into this file.
import os
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN") or ""   # e.g. "hf_..."
assert os.environ["HF_TOKEN"], "Set the HF_TOKEN environment variable"

# %% ---------------------------------------------------------------------------

# 1. Settings and helpers (reused from train_v3 so crops and embeddings match training-time evaluation)
import base64, io, json, random, sys
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

OUT_DIR = "decore/evals/" + \
         datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
MODEL_REPO = "AlphJain/home-decor-siglip"
# Models: "base"; "<run>" = HF run's best checkpoint; "<run>@<step>" = HF run at that step;
# "run:<run>" = the data folder run's best checkpoint (decore/models/<run>/). Differences: each model vs base and vs RUNS[0].
# v3-baseline is kept so its 500 STL selection queries stay excluded (same 2,250 held-out queries as before);
# differences are reported vs base and vs RUNS[0] (the current model, v3-clean-nostyle)
RUNS = ["run:v3-clean-nostyle", "run:v3-clean-oi", "v3-baseline"]
MISS_MODEL = "run:v3-clean-oi"              # whose mistakes to show
MODELS_DIR = "decore/models"
TEST_SPLIT = "test"                        # natural-language test (cell 4b): never used to pick checkpoints
ABO_TAG_RUN_HELDOUT = ""                   # openai_label_jobs Part 2a run (validation/test): ABO main photos to skip
N_MISSES, BOOTSTRAP, SEED = 60, 1000, 0
token = os.environ["HF_TOKEN"]
t.TOKEN = token
assert torch.cuda.is_available(), "A GPU is required"
device, dtype = torch.device("cuda"), torch.bfloat16
print("GPU:", torch.cuda.get_device_name(0), "| results:", OUT_DIR)

# %% ---------------------------------------------------------------------------

# 2. STL data: held-out queries (not used for checkpoint selection) and the full gallery
root = t.ensure_stl()
queries = [json.loads(x) for x in (root / "queries.jsonl").read_text().splitlines() if x]
gallery = [json.loads(x) for x in (root / "gallery.jsonl").read_text().splitlines() if x]
used = set()   # STL queries any HF run used to pick its checkpoint (Local runs use none)
for run in {r.split("@")[0] for r in RUNS if not r.startswith("run:")}:
    used |= set(json.loads(Path(hf_hub_download(MODEL_REPO, f"runs/{run}/stl_selection_queries.json",
                                                token=token)).read_text())["positions_in_queries_jsonl"])
held = [q for n, q in enumerate(queries) if n not in used]
index = {g["product"]: n for n, g in enumerate(gallery)}
relevant = [{index[p] for p in q["relevant"] if p in index} for q in held]
categories = np.array([q["category"] for q in held])
print(f"{len(queries)} queries, {len(used)} used for selection, {len(held)} held out; gallery {len(gallery)}")


def crop(q):   # identical to train_v3.stl_eval
    def load():
        l, tp, r, b = q["bbox"]
        need = int(2 * 224 / max(min(r - l, b - tp), 1e-3))
        image = t.open_scaled(root / "scenes" / f"{q['scene']}.jpg", min(max(need, 448), 4096))
        w, h = image.size
        px, py = (r - l) * .1, (b - tp) * .1
        return image.crop((max(0, (l - px) * w), max(0, (tp - py) * h),
                           min(w, (r + px) * w), min(h, (b + py) * h)))
    return load


def product(sig):
    return lambda: t.open_scaled(root / "products" / f"{sig}.jpg", 448)

# %% ---------------------------------------------------------------------------

# 3. Embed and rank with each model


def read_file(path):
    return fs.head(path, fs.ls(path)[0].size)


def stored_model_dir(run, step=None):
    # Stored checkpoints are base64 parts + manifest per file, listed in files.json (train_v3.Store)
    root = f"{MODELS_DIR}/{run}"
    step = step or json.loads(read_file(f"{root}/best.json"))["step"]
    rel = f"{root}/checkpoints/step-{int(step):08d}/model"
    local = Path(f"/tmp/decore/eval_models/{run}/step-{int(step):08d}")
    for name in json.loads(read_file(f"{rel}/files.json")):
        if (local / name).exists():
            continue
        manifest = json.loads(read_file(f"{rel}/{name}/manifest.json"))
        data = b"".join(base64.b64decode(read_file(f"{rel}/{name}/{p}")) for p in manifest["parts"])
        import hashlib
        assert hashlib.sha256(data).hexdigest() == manifest["sha256"], f"checksum mismatch {name}"
        (local / name).parent.mkdir(parents=True, exist_ok=True)
        (local / name).write_bytes(data)
    return local, step


def load_model(name):
    if name == "base":
        source = t.BASE_MODEL
    elif name.startswith("run:"):
        source, step = stored_model_dir(name[4:])
        print(name, "best step", step)
    else:
        run, _, step = name.partition("@")
        step = int(step) if step else json.loads(Path(hf_hub_download(MODEL_REPO, f"runs/{run}/best.json", token=token,
                                                                      force_download=True)).read_text())["step"]
        prefix = f"runs/{run}/checkpoints/step-{step:08d}/model"
        source = str(Path(snapshot_download(MODEL_REPO, token=token, allow_patterns=[f"{prefix}/*"])) / prefix)
        print(name, "step", step)
    return (AutoModel.from_pretrained(source, dtype=torch.float32).to(device).eval(),
            AutoProcessor.from_pretrained(source))


def first_relevant_rank(sim):
    ranks = []
    for n, rel in enumerate(relevant):
        best = sim[n, torch.tensor(sorted(rel))].max()
        ranks.append(int((sim[n] > best).sum()) + 1)
    return np.asarray(ranks)


ranks, top3, gallery_emb = {}, {}, {}
for name in ["base"] + RUNS:
    model, processor = load_model(name)
    qe = t.embed_images(model, processor, [crop(q) for q in held], device, dtype)
    ge = t.embed_images(model, processor, [product(g["product"]) for g in gallery], device, dtype)
    gallery_emb[name] = ge   # kept for the text-search test (cell 8)
    sim = qe @ ge.T
    ranks[name] = first_relevant_rank(sim)
    top3[name] = torch.topk(sim, 3, dim=1).indices.numpy()
    top10 = torch.topk(sim, 10, dim=1).indices.numpy()   # saved for the judge (openai_label_jobs Part 1)
    assert fs.put(f"{OUT_DIR}/stl_top10/{name.replace(':', '_').replace('@', '_')}.jsonl", "\n".join(
        json.dumps({"scene": q["scene"], "bbox": q["bbox"], "truth": sorted(q["relevant"]),
                    "predicted": [gallery[int(i)]["product"] for i in top10[n]]}) for n, q in enumerate(held)),
        overwrite=True)
    del model; torch.cuda.empty_cache()
    print(name, {k: round(float((ranks[name] <= k).mean()), 4) for k in (1, 10, 50)}, flush=True)

# %% ---------------------------------------------------------------------------

# 4. Results: overall, per category, and bootstrap intervals for differences


def metrics(r):
    return {"R@1": float((r <= 1).mean()), "R@10": float((r <= 10).mean()), "R@50": float((r <= 50).mean()),
            "MRR": float((1 / r).mean()), "median_rank": float(np.median(r)), "queries": int(len(r))}


def bootstrap_diff(a, b, k):
    # 95% interval for (model a - model b) on the same queries, resampling queries
    rng = np.random.default_rng(SEED)
    hit_a, hit_b = (ranks[a] <= k).astype(float), (ranks[b] <= k).astype(float)
    idx = rng.integers(0, len(hit_a), size=(BOOTSTRAP, len(hit_a)))
    d = hit_a[idx].mean(1) - hit_b[idx].mean(1)
    return {"diff": float(hit_a.mean() - hit_b.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


results = {"held_out_queries": len(held), "gallery": len(gallery),
           "overall": {name: metrics(r) for name, r in ranks.items()},
           "per_category": {}, "differences": {}}
for c in sorted(set(categories)):
    mask = categories == c
    results["per_category"][c] = {"queries": int(mask.sum()),
                                  **{name: {"R@1": round(float((r[mask] <= 1).mean()), 3),
                                            "R@10": round(float((r[mask] <= 10).mean()), 3)} for name, r in ranks.items()}}
for a, b in [(r, "base") for r in RUNS] + [(r, RUNS[0]) for r in RUNS[1:]]:
    for k in (1, 10):
        results["differences"][f"{a} - {b} R@{k}"] = bootstrap_diff(a, b, k)

print(json.dumps(results["overall"], indent=1))
print(json.dumps(results["differences"], indent=1))
print(f"{'category':16s} {'n':>5s} " + " ".join(f"{n[:12]:>16s}" for n in ranks))
for c, row in results["per_category"].items():
    print(f"{c:16s} {row['queries']:5d} " + " ".join(f"{row[n]['R@1']:7.3f} /{row[n]['R@10']:6.3f}" for n in ranks))
assert fs.put(f"{OUT_DIR}/results.json", json.dumps(results, indent=2), overwrite=True)

# %% ---------------------------------------------------------------------------

# 4b. Natural-language search on HELD-OUT images (test split: never trained on, never used to pick checkpoints).
# Queries are the new-style texts written for the test images by data/recaption.py (SPLIT="test"):
#   scenes: caption -> room image, query -> room image   (gallery = all test scenes)
#   crops:  caption -> crop image,  query -> crop image   (gallery = all test crops, wrong items removed)
#   ABO:    query -> product photo                        (gallery = main photo of every test product)
# One correct image per query, so generic queries ("modern living room") are hard for every model alike;
# compare models with each other, not with 100%.
# TEST split (validation is now used to pick checkpoints): images and text never used for training or selection
SRC_VAL = "decore/sources/home-decore/20261104T130155Z"
RECAP_VAL = f"decore/checks/recaption/{TEST_SPLIT}_v1"


def read_jsonl(path):
    return [json.loads(l) for l in fs.head(path, fs.ls(path)[0].size).splitlines() if l]


def read_folder(folder):
    return [r for p in sorted(f.path for f in fs.ls(folder) if f.name.endswith(".jsonl")) for r in read_jsonl(p)]


recap = {r["key"]: r for c in ("abo", "deepfurniture_scenes", "deepfurniture_crops") for r in read_folder(f"{RECAP_VAL}/{c}")}
nl_tests = {}   # name -> (gallery image b64 list, [(query text, gallery index)])
for config, short in (("deepfurniture_scenes", "scenes"), ("deepfurniture_crops", "crops")):
    rows = []
    for r in read_folder(f"{SRC_VAL}/{config}/{TEST_SPLIT}"):
        rec = recap.get(f"{config}/{TEST_SPLIT}:{r['row_index']}")
        # same label-quality rule as train / selection: crops too small or cut off to show the item are skipped
        if (r.get("area_ratio") or 1.0) < 0.01 or (r.get("in_image_ratio") or 1.0) < 0.8:
            continue
        if rec and rec.get("rc_caption") and not rec.get("rc_wrong_item"):
            rows.append((r["image_b64"], rec))
    images = [b for b, _ in rows]   # one list object, so the caption and query tests share one gallery embedding
    nl_tests[f"{short}: caption"] = (images, [(rec["rc_caption"], i) for i, (_, rec) in enumerate(rows)])
    nl_tests[f"{short}: query"] = (images, [(q, i) for i, (_, rec) in enumerate(rows) for q in rec.get("rc_queries", [])])
# the test split was never cleaned: skip colour-swatch listings (the photo is a fabric square, the query describes the sofa)
assert ABO_TAG_RUN_HELDOUT, "Set ABO_TAG_RUN_HELDOUT"
abo_skip = set(fs.head(p := f"decore/checks/clean/"
                                     f"{ABO_TAG_RUN_HELDOUT}/abo_{TEST_SPLIT}/exclude.txt", fs.ls(p)[0].size).split())
print(f"ABO {TEST_SPLIT} main photos skipped (swatch / placeholder): {len(abo_skip)}")
mains = [r for r in read_folder(f"{SRC_VAL}/abo/{TEST_SPLIT}") if r.get("is_main") and recap.get(f"abo/group:{r['group_id']}")
         and f"abo/{TEST_SPLIT}:{r['row_index']}" not in abo_skip
         and "swatch" not in f"{r.get('title', '')} {r.get('raw_title', '')}".lower()]
nl_tests["abo: query"] = ([r["image_b64"] for r in mains],
                          [(q, i) for i, r in enumerate(mains) for q in recap[f"abo/group:{r['group_id']}"]["rc_queries"]])
for name, (g, q) in nl_tests.items():
    print(f"{name:16s} gallery {len(g):6,} | queries {len(q):6,}")

nl_ranks = {name: {} for name in nl_tests}
for model_name in ["base"] + RUNS:
    model, processor = load_model(model_name)
    gallery_cache = {}
    for name, (images, pairs) in nl_tests.items():
        key = id(images)
        if key not in gallery_cache:   # caption and query tests of one source share the gallery
            gallery_cache[key] = t.embed_images(model, processor,
                                                [lambda b=b: {"bytes": base64.b64decode(b)} for b in images], device, dtype)
        ge = gallery_cache[key]
        qe = t.embed_texts(model, processor, [text for text, _ in pairs], device, dtype)
        target = torch.tensor([i for _, i in pairs])
        sim = qe @ ge.T
        own = sim[torch.arange(len(pairs)), target]
        nl_ranks[name][model_name] = ((sim > own[:, None]).sum(1) + 1).numpy()
    del model; torch.cuda.empty_cache()
    print(model_name, {n: round(float((r[model_name] <= 10).mean()), 3) for n, r in nl_ranks.items()}, flush=True)


def nl_diff(name, a, b, k):
    rng = np.random.default_rng(SEED)
    ha, hb = (nl_ranks[name][a] <= k).astype(float), (nl_ranks[name][b] <= k).astype(float)
    idx = rng.integers(0, len(ha), size=(BOOTSTRAP, len(ha)))
    d = ha[idx].mean(1) - hb[idx].mean(1)
    return {"diff": round(float(ha.mean() - hb.mean()), 4),
            "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]}


nl_results = {name: {"queries": len(nl_tests[name][1]), "gallery": len(nl_tests[name][0]),
                     **{m: {"R@1": round(float((r <= 1).mean()), 4), "R@10": round(float((r <= 10).mean()), 4),
                            "MRR": round(float((1 / r).mean()), 4)} for m, r in nl_ranks[name].items()},
                     **{f"{r} - {RUNS[0]} R@10": nl_diff(name, r, RUNS[0], 10) for r in RUNS[1:]},
                     **{f"{r} - base R@10": nl_diff(name, r, "base", 10) for r in RUNS}}
              for name in nl_tests}
print(json.dumps(nl_results, indent=1))
assert fs.put(f"{OUT_DIR}/results_natural_language.json", json.dumps(nl_results, indent=2), overwrite=True)

# %% ---------------------------------------------------------------------------

# 5. Gallery of mistakes: query crop | correct product | model's top 3 (wrong at rank 1)


def thumb(image, side=180):
    image = image.copy(); image.thumbnail((side, side))
    buffer = io.BytesIO(); image.save(buffer, format="JPEG", quality=85)
    return f"<img src='data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}'>"


misses = [n for n in range(len(held)) if ranks[MISS_MODEL][n] > 1]
pick = sorted(random.Random(SEED).sample(misses, min(N_MISSES, len(misses))))
rows = []
for n in pick:
    q = held[n]
    cells = [thumb(crop(q)()), thumb(product(q["product"])())]
    cells += [thumb(product(gallery[i]["product"])()) for i in top3[MISS_MODEL][n]]
    rows.append(f"<tr><td>{q['category']}<br>rank of correct: {ranks[MISS_MODEL][n]}<br>"
                f"base rank: {ranks['base'][n]}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
html = (f"<html><body><h3>{MISS_MODEL}: {len(pick)} random rank-1 misses out of {len(misses)} "
        f"(held-out STL)</h3><table border=1 cellpadding=3><tr><th>info</th><th>query crop</th>"
        f"<th>correct product</th><th>top 1</th><th>top 2</th><th>top 3</th></tr>{''.join(rows)}</table>"
        f"<p>For each row, note one: look-alike (same design, other colour/seller) / right type, wrong product / "
        f"wrong type / bad crop / label looks wrong.</p></body></html>")
assert fs.put(f"{OUT_DIR}/misses_{MISS_MODEL.replace(':', '_')}.html", html, overwrite=True)
show_html(html)
print("saved:", OUT_DIR)

# %% ---------------------------------------------------------------------------

# 6. Blind side-by-side: base vs trained, judged by eye, no STL labels involved.
# For each random held-out query, one model is shown as A and the other as B (random per query).
# Write down A, B or "same" for each row (which top 3 matches the photo better), then run cell 7.
COMPARE_MODEL, N_COMPARE = "v3-baseline", 50
rng = random.Random(SEED + 1)
compare = sorted(rng.sample(range(len(held)), N_COMPARE))
key, rows = {}, []
for number, n in enumerate(compare, 1):
    a, b = ("base", COMPARE_MODEL) if rng.random() < 0.5 else (COMPARE_MODEL, "base")
    key[number] = {"A": a, "B": b, "query_index": n}
    q = held[n]
    sides = ["".join(f"<td>{thumb(product(gallery[i]['product'])(), 150)}</td>" for i in top3[m][n]) for m in (a, b)]
    rows.append(f"<tr><td><b>#{number}</b><br>{q['category']}</td><td>{thumb(crop(q)(), 170)}</td>"
                f"<td style='background:#eef'>A</td>{sides[0]}<td style='background:#fee'>B</td>{sides[1]}</tr>")
compare_html = (f"<html><body><h3>Blind comparison: {N_COMPARE} random held-out queries</h3>"
                f"<p>For each row: which top 3 (A or B) matches the query photo better? Answer A, B or same.</p>"
                f"<table border=1 cellpadding=3><tr><th>#</th><th>query</th><th></th><th colspan=3>A: top 3</th>"
                f"<th></th><th colspan=3>B: top 3</th></tr>{''.join(rows)}</table></body></html>")
assert fs.put(f"{OUT_DIR}/blind_compare.html", compare_html, overwrite=True)
assert fs.put(f"{OUT_DIR}/blind_compare_key.json", json.dumps(key, indent=2), overwrite=True)
show_html(compare_html)

# %% ---------------------------------------------------------------------------

# 7. Reveal: paste your answers in order (A / B / same, one per row) and count wins per model.
ANSWERS = ""   # e.g. "A B B same A ..." (50 entries)
answers = ANSWERS.split()
assert len(answers) == len(key), f"need {len(key)} answers, got {len(answers)}"
wins = {"base": 0, COMPARE_MODEL: 0, "same": 0}
for number, answer in enumerate(answers, 1):
    answer = answer.strip().lower()
    wins["same" if answer == "same" else key[number]["A" if answer == "a" else "B"]] += 1
print(wins)

# %% ---------------------------------------------------------------------------

# 8. Blind text search on real products: shopping phrases -> STL product gallery, base vs trained.
# STL has no text queries, so these phrases are the test. Judge each row like cell 6: which top 5 fits the
# phrase better (A, B or same)? Then run cell 9.
TEXT_QUERIES = [
    "cognac leather sofa", "grey tufted sectional sofa", "white slipcovered sofa", "navy velvet sofa",
    "mid-century modern sofa with wooden legs", "beige linen loveseat", "green velvet armchair",
    "tufted wingback chair", "white molded plastic dining chair with wooden legs", "rattan accent chair",
    "black metal cafe chair", "leather club chair", "wooden saddle bar stool with metal legs",
    "white counter stool", "industrial metal bar stool", "brass floor lamp", "white ceramic table lamp",
    "arc floor lamp", "rattan pendant light", "galvanized barn pendant light", "crystal chandelier",
    "black metal cage pendant", "clear glass globe pendant", "moroccan shag rug", "blue persian rug",
    "jute area rug", "black and white striped rug", "colorful bohemian rug", "white sheer curtains",
    "blackout curtains in dark grey", "linen curtains in natural beige", "velvet curtains emerald green",
    "round gold mirror", "sunburst mirror", "full length floor mirror with black frame", "arched wall mirror",
    "blue and white throw pillow", "mustard yellow velvet pillow", "faux fur throw pillow",
    "geometric black and white pillow", "brushed gold kitchen faucet", "matte black bathroom faucet",
    "chrome pull-down kitchen faucet", "ceramic vase with dried flowers", "woven seagrass basket",
    "marble coffee table", "round wooden side table", "glass console table", "boho living room decor",
    "scandinavian minimalist chair", "japandi wooden stool", "farmhouse pendant light", "coastal blue rug",
    "art deco brass mirror", "industrial style floor lamp", "traditional rolled arm sofa",
    "modern glam velvet chair", "rustic wooden bench", "minimalist white pendant", "vintage leather armchair",
]
text_top = {}
for name in ("base", COMPARE_MODEL):
    model, processor = load_model(name)
    te = t.embed_texts(model, processor, TEXT_QUERIES, device, dtype)
    text_top[name] = torch.topk(te @ gallery_emb[name].T, 5, dim=1).indices.numpy()
    del model; torch.cuda.empty_cache()

rng = random.Random(SEED + 2)
text_key, rows = {}, []
for number, phrase in enumerate(TEXT_QUERIES, 1):
    a, b = ("base", COMPARE_MODEL) if rng.random() < 0.5 else (COMPARE_MODEL, "base")
    text_key[number] = {"A": a, "B": b, "query": phrase}
    sides = ["".join(f"<td>{thumb(product(gallery[i]['product'])(), 120)}</td>"
                     for i in text_top[m][number - 1]) for m in (a, b)]
    rows.append(f"<tr><td><b>#{number}</b><br>{phrase}</td><td style='background:#eef'>A</td>{sides[0]}"
                f"<td style='background:#fee'>B</td>{sides[1]}</tr>")
text_html = (f"<html><body><h3>Blind text search: {len(TEXT_QUERIES)} phrases over the STL product gallery</h3>"
             f"<p>For each row: which top 5 (A or B) fits the phrase better? Answer A, B or same.</p>"
             f"<table border=1 cellpadding=3><tr><th>phrase</th><th></th><th colspan=5>A: top 5</th><th></th>"
             f"<th colspan=5>B: top 5</th></tr>{''.join(rows)}</table></body></html>")
assert fs.put(f"{OUT_DIR}/blind_text.html", text_html, overwrite=True)
assert fs.put(f"{OUT_DIR}/blind_text_key.json", json.dumps(text_key, indent=2), overwrite=True)
show_html(text_html)

# %% ---------------------------------------------------------------------------

# 9. Reveal the text-search comparison
TEXT_ANSWERS = ""   # e.g. "A same B ..." (one per phrase)
answers = TEXT_ANSWERS.split()
assert len(answers) == len(text_key), f"need {len(text_key)} answers, got {len(answers)}"
wins = {"base": 0, COMPARE_MODEL: 0, "same": 0}
for number, answer in enumerate(answers, 1):
    answer = answer.strip().lower()
    wins["same" if answer == "same" else text_key[number]["A" if answer == "a" else "B"]] += 1
print(wins)

# %% ---------------------------------------------------------------------------

# 10. Training-data check (reads the training data from the data folder; no model needed).
# Page 1: random rows per source with the image and every text field -> mark fields right / vague / wrong.
# Page 2: random training pairs (DF crop -> product render, ABO view -> view, Pix3D photo -> photo) -> same object?
from PIL import Image as PILImage

DATA_DIR = "decore/sources"
SOURCE_RUNS = {"home-decore": "20261104T130155Z", "pix3d": "20261104T151426Z", "coco_scenes": "20261104T151655Z"}
ROWS_PER_SOURCE, PAIRS_PER_KIND = 15, 12
TEXT_FIELDS = ["title", "vlm_caption", "captions", "category", "category_coarse", "style", "vlm_style",
               "material", "vlm_materials", "color", "vlm_colors", "keywords", "vlm_details"]


def read_text_file(path):
    return fs.head(path, fs.ls(path)[0].size)


def read_rows(folder, shards=2):
    paths = sorted(f.path for f in fs.ls(folder) if f.name.endswith(".jsonl"))
    picked = random.Random(SEED).sample(paths, min(shards, len(paths)))
    return [json.loads(l) for p in picked for l in read_text_file(p).splitlines() if l]


def b64_image(value):
    return PILImage.open(io.BytesIO(base64.b64decode(value))).convert("RGB")


folders = {
    "abo": f"{DATA_DIR}/home-decore/{SOURCE_RUNS['home-decore']}/abo/train",
    "deepfurniture_crops": f"{DATA_DIR}/home-decore/{SOURCE_RUNS['home-decore']}/deepfurniture_crops/train",
    "deepfurniture_scenes": f"{DATA_DIR}/home-decore/{SOURCE_RUNS['home-decore']}/deepfurniture_scenes/train",
    "pix3d": f"{DATA_DIR}/pix3d/{SOURCE_RUNS['pix3d']}/train",
    "coco_scenes": f"{DATA_DIR}/coco_scenes/{SOURCE_RUNS['coco_scenes']}/train",
}
loaded = {source: read_rows(folder) for source, folder in folders.items()}

rows = []
for source, data in loaded.items():
    for r in random.Random(SEED).sample(data, min(ROWS_PER_SOURCE, len(data))):
        fields = "".join(f"<b>{f}</b>: {r[f]}<br>" for f in TEXT_FIELDS if r.get(f) not in (None, "", [], "none"))
        rows.append(f"<tr><td>{source}</td><td>{thumb(b64_image(r['image_b64']), 200)}</td>"
                    f"<td style='font-size:12px'>{fields}</td></tr>")
fields_html = (f"<html><body><h3>Training rows: image + text fields</h3><p>Per field: right / vague / wrong.</p>"
               f"<table border=1 cellpadding=3>{''.join(rows)}</table></body></html>")
assert fs.put(f"{OUT_DIR}/data_check_fields.html", fields_html, overwrite=True)
show_html(fields_html)

# %% ---------------------------------------------------------------------------

# 11. Training pairs: are both images the same object?
pairs = []
crops = random.Random(SEED).sample(loaded["deepfurniture_crops"], PAIRS_PER_KIND)
wanted = {str(c["furniture_id"]) for c in crops}
renders = {}
for path in sorted(f.path for f in fs.ls(folders["deepfurniture_crops"].replace("crops", "products"))
                   if f.name.endswith(".jsonl")):
    for line in read_text_file(path).splitlines():
        r = json.loads(line) if line else {}
        if str(r.get("furniture_id")) in wanted:
            renders[str(r["furniture_id"])] = r["image_b64"]
    if len(renders) == len(wanted):
        break
for c in crops:
    if str(c["furniture_id"]) in renders:
        pairs.append(("DF crop -> render", c["image_b64"], renders[str(c["furniture_id"])], c["furniture_id"]))
for source, label in (("abo", "ABO view -> view"), ("pix3d", "Pix3D photo -> photo")):
    groups = {}
    for r in loaded[source]:
        groups.setdefault(r["group_id"], []).append(r)
    multi = [g for g in groups.values() if len(g) > 1]
    for g in random.Random(SEED).sample(multi, min(PAIRS_PER_KIND, len(multi))):
        pairs.append((label, g[0]["image_b64"], g[1]["image_b64"], g[0]["group_id"]))
rows = [f"<tr><td>{kind}<br><small>{gid}</small></td><td>{thumb(b64_image(a), 200)}</td>"
        f"<td>{thumb(b64_image(b), 200)}</td></tr>" for kind, a, b, gid in pairs]
pairs_html = (f"<html><body><h3>Training pairs: same object?</h3><table border=1 cellpadding=3>{''.join(rows)}"
              f"</table></body></html>")
assert fs.put(f"{OUT_DIR}/data_check_pairs.html", pairs_html, overwrite=True)
show_html(pairs_html)

# %% ---------------------------------------------------------------------------

# 12. Noise counts on all ABO train rows: keywords naming another category, seller vs VLM material/colour.
CATEGORY_WORDS = ["sofa", "couch", "chair", "stool", "bed", "lamp", "rug", "table", "curtain", "mirror",
                  "pillow", "cabinet", "shelf", "desk", "dresser", "vase", "basket", "clock", "bench", "ottoman"]
abo = [json.loads(l) for p in sorted(f.path for f in fs.ls(folders["abo"]) if f.name.endswith(".jsonl"))
       for l in read_text_file(p).splitlines() if l]


def words(value):
    return {w.strip().lower() for w in str(value or "").replace(",", " ").split() if w.strip()}


def other_category(r):
    own = words(r.get("category")) | words(r.get("category_coarse"))
    return bool({w for w in words(r.get("keywords")) if w.rstrip("s") in CATEGORY_WORDS} - own
                - {w + "s" for w in own})


def disagree(seller, vlm):
    a, b = words(seller), words(vlm)
    return bool(a and b and not (a & b))


has_kw = [r for r in abo if r.get("keywords")]
has_mat = [r for r in abo if r.get("material") and r.get("vlm_materials")]
has_col = [r for r in abo if r.get("color") and r.get("vlm_colors")]
noise = {
    "abo_rows": len(abo),
    "keywords_present": len(has_kw),
    "keywords_naming_another_category": round(sum(map(other_category, has_kw)) / max(1, len(has_kw)), 3),
    "material_seller_vs_vlm_no_overlap": round(sum(disagree(r["material"], r["vlm_materials"]) for r in has_mat)
                                               / max(1, len(has_mat)), 3),
    "color_seller_vs_vlm_no_overlap": round(sum(disagree(r["color"], r["vlm_colors"]) for r in has_col)
                                            / max(1, len(has_col)), 3),
}
print(json.dumps(noise, indent=2))
assert fs.put(f"{OUT_DIR}/data_check_noise.json", json.dumps(noise, indent=2), overwrite=True)
print("saved:", OUT_DIR)
