# # Leak check: all training data in the data folder vs. the eval sets
#
# Step 2 of 3. Run after `export_existing`, `pix3d` and `coco` have saved their data to the data folder.
#
# Why: STL-Home (our real-photo test) comes from Pinterest, which re-posts web photos: Amazon product shots
# (like ABO), Flickr photos (like COCO), IKEA pages (like Pix3D). If the same picture is in training and in
# the test, the model has seen the answer and the STL score is inflated. This finds those pictures.
#
# For each source it reads the training rows back from the data folder, compares every image with every eval image
# (STL scenes, query crops, products; home-decore validation + test) and saves a report and an
# `exclude.txt` per source. The data in the data folder is not changed; the HF upload (step 3) skips the excluded rows.
#
# Needs `leak_check.py` (plain file) in the same folder, an HF read token (to load the private eval sets),
# and a GPU (~200k images to embed).


import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "finetuning")]
from fsutil import fs, show_html   # file helpers and HTML review pages
# %% ---------------------------------------------------------------------------

# HF token: read access to AlphJain/home-decore and AlphJain/stl-home-eval.
# Set it as an environment variable before running; never write it into this file.
import os
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN") or ""   # e.g. "hf_..."
assert os.environ["HF_TOKEN"], "Set the HF_TOKEN environment variable"

# %% ---------------------------------------------------------------------------

import os, sys
sys.path.insert(0, HERE)   # leak_check.py is a plain file next to this script
from leak_check import *

# %% ---------------------------------------------------------------------------

# 1. Settings
import base64, json
from datetime import datetime, timezone
from functools import lru_cache

DATA_ROOT = "decore"
# Source -> run folder name in the data folder (None = the latest run). What gets checked per source:
#   new sources: train + validation (their own validation must not overlap the test either)
#   home-decore: train only (its validation/test are part of the eval index)
RUNS = {"home-decore": None, "pix3d": None, "coco_scenes": None}
LIMIT_SHARDS = None        # e.g. 2 for a quick first run; None = everything

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
check_out = f"{DATA_ROOT}/checks/leak/{RUN_ID}"


def latest(source):
    runs = sorted(f.name.rstrip("/") for f in fs.ls(f"{DATA_ROOT}/sources/{source}/") if f.isDir())
    return runs[-1]


runs = {source: run or latest(source) for source, run in RUNS.items()}
print("device:", DEVICE, "| checking:", runs, "| report:", check_out)

# %% ---------------------------------------------------------------------------

# 2. Find the shards to check


def shard_paths(folder):
    return sorted(f.path for f in fs.ls(folder) if f.name.endswith(".jsonl"))[:LIMIT_SHARDS]


shards = {}
for source, run in runs.items():
    base = f"{DATA_ROOT}/sources/{source}/{run}"
    if source == "home-decore":
        configs = [f.name.rstrip("/") for f in fs.ls(base) if f.isDir()]
        shards[source] = [p for c in sorted(configs) for p in shard_paths(f"{base}/{c}/train")]
    else:
        shards[source] = [p for split in ("train", "validation") for p in shard_paths(f"{base}/{split}")]
    print(source, len(shards[source]), "shards")


@lru_cache(maxsize=4)
def read_shard(path):
    size = fs.ls(path)[0].size
    return [json.loads(line) for line in fs.head(path, size).splitlines() if line]


def row_key(source, path, row):
    # Stable id used in exclude.txt: image_id for new sources, config/split:row_index for home-decore.
    if source == "home-decore":
        config, split = path.split("/")[-3:-1]
        return f"{config}/{split}:{row['row_index']}"
    return row["image_id"]

# %% ---------------------------------------------------------------------------

# 3. Fingerprint shard by shard, then compare with the eval index and save a report per source
where = {}   # name -> (shard path, line, image field), to show training images in the contact sheet


def load_train(name):
    path, line, field = where[name]
    return to_pil(base64.b64decode(read_shard(path)[line][field]))


overall = {}
for source, paths in shards.items():
    parts = []
    for path in paths:
        rows = read_shard(path)
        loaders, names = [], []
        for line, row in enumerate(rows):
            key = row_key(source, path, row)
            fields = [("image_b64", "image")] + ([("source_thumb_b64", "photo")] if "source_thumb_b64" in row else [])
            for field, kind in fields:
                name = f"{key}#{kind}"
                where[name] = (path, line, field)
                loaders.append(lambda b=row[field]: to_pil(base64.b64decode(b)))
                names.append(name)
        parts.append(fingerprint(loaders, names))
    train = merge(parts)
    keep, summary, rows_text, html = check(train, source, load_train)

    # A row is excluded if any of its images (crop or full photo) matched an eval image.
    exclude = sorted({name.split("#")[0] for name, k in zip(train["names"], keep) if not k})
    summary["rows_excluded"] = len(exclude)
    summary["run"] = runs[source]
    overall[source] = summary
    for name, text in (("leak_summary.json", json.dumps(summary, indent=2)), ("leak_rows.jsonl", rows_text),
                       ("leak_top.html", html), ("exclude.txt", "\n".join(exclude) + "\n")):
        assert fs.put(f"{check_out}/{source}/{name}", text, overwrite=True), name
    print(f"{source}: {len(exclude)} rows to exclude")
    show_html(html)

# %% ---------------------------------------------------------------------------

# 4. Overall summary
assert fs.put(f"{check_out}/summary.json", json.dumps(overall, indent=2), overwrite=True)
print(json.dumps({s: {k: v[k] for k in ("checked", "dropped", "rows_excluded", "dropped_by_eval_set")}
                  for s, v in overall.items()}, indent=2))
print("saved:", check_out)
