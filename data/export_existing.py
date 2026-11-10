# # Existing data: copy AlphJain/home-decore to the data folder
#
# Step 1 of 3 (with `pix3d` and `coco`): copies every split of the four existing configs to the data folder in the same
# JSONL format, so all data lives in one place. Original image bytes are kept (not re-encoded).
# Nothing is filtered here; the leak check (`run_leak_check`) runs afterwards on all sources.
#
# Needs an HF read token (the repo is private). No GPU needed. About 4-5 GB of images (~6 GB as base64 text).


import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "finetuning")]
from fsutil import fs, show_html   # file helpers and HTML review pages
# %% ---------------------------------------------------------------------------

# HF token: read access to AlphJain/home-decore.
# Set it as an environment variable before running; never write it into this file.
import os
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN") or ""   # e.g. "hf_..."
assert os.environ["HF_TOKEN"], "Set the HF_TOKEN environment variable"

# %% ---------------------------------------------------------------------------

# 1. Settings
import base64, json
from datetime import datetime, timezone
from pathlib import Path

from datasets import Image as ImageFeature, load_dataset

DATA_ROOT = "decore"
DATA_REPO = "AlphJain/home-decore"
CONFIGS = ["abo", "deepfurniture_crops", "deepfurniture_products", "deepfurniture_scenes"]
SPLITS = ["train", "validation", "test"]
SHARD_MB = 40              # target size of one JSONL file on disk (the preflight below tests 50 MB)
WORK = Path("/local_disk0/decore" if Path("/local_disk0").exists() else "/tmp/decore")   # HF download cache

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
run_out = f"{DATA_ROOT}/sources/home-decore/{RUN_ID}"
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
assert same, "Large text round trip failed; lower SHARD_MB and the probe size"

# %% ---------------------------------------------------------------------------

# 3. Copy every config and split. Row order is kept, so `row_index` matches the HF split.
manifest = {"source": DATA_REPO, "run_id": RUN_ID, "run_out": run_out,
            "format": "JSONL; one row per image; original image bytes as base64 in `image_b64`; "
                      "`row_index` = position in the HF split",
            "license": "abo: CC BY-NC 4.0 (paper) / CC BY 4.0 (dataset page), confirm; deepfurniture: see dataset card",
            "configs": {}}
for config in CONFIGS:
    manifest["configs"][config] = {}
    for split in SPLITS:
        ds = load_dataset(DATA_REPO, config, split=split, token=os.environ["HF_TOKEN"],
                          cache_dir=str(WORK / "hf_cache"))
        column = next(c for c in ("image", "crop", "scene", "render") if c in ds.column_names)
        ds = ds.cast_column(column, ImageFeature(decode=False))
        shards, lines, size = [], [], 0
        for start in range(0, len(ds), 1000):
            batch = ds[start:start + 1000]
            for k in range(len(batch[column])):
                row = {c: batch[c][k] for c in ds.column_names if c != column}
                row["row_index"] = start + k
                row["image_b64"] = base64.b64encode(batch[column][k]["bytes"]).decode()
                line = json.dumps(row)
                lines.append(line); size += len(line) + 1
                if size >= SHARD_MB * 1_000_000 or (start + k == len(ds) - 1):
                    path = f"{run_out}/{config}/{split}/part-{len(shards):05d}.jsonl"
                    assert fs.put(path, "\n".join(lines) + "\n", overwrite=True), path
                    shards.append(path); lines, size = [], 0
        manifest["configs"][config][split] = {"rows": len(ds), "shards": len(shards)}
        print(f"{config}/{split}: {len(ds)} rows -> {len(shards)} shards", flush=True)

assert fs.put(f"{run_out}/manifest.json", json.dumps(manifest, indent=2), overwrite=True)
print("saved:", run_out)
print([f.path for f in fs.ls(run_out)])
