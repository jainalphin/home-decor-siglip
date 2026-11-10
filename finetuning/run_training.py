# # Home-decor SigLIP 2: training (v3)
#
# Trains with the data in the data folder: leak-checked existing data, plus optionally Pix3D (real furniture photos)
# and COCO (real rooms). Two experiments, same recipe (Run B's settings), same number of steps:
#
# | `EXPERIMENT` | data | question |
# |---|---|---|
# | `baseline` | existing data, leak-checked | Run B again, on clean data and to the end (Run B stopped at 750) |
# | `real_photos` | + `pix3d` + `coco_scenes` | do real photos improve STL? |
# | `clean` | existing data after `data/build_clean_dataset.py` (train only) | does cleaning beat `baseline`? |
#
# Needs `train_v3.py` in the same folder, a GPU, and an HF token with read access to the base dataset and the STL
# eval set. `EXPERIMENT = "clean"` (the final model) saves checkpoints under decore/models/<run>/; the older
# experiments save to a private HF model repo. Run with `DRY_RUN = True` first (CPU, no training).


import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "finetuning")]
from fsutil import fs, show_html   # file helpers and HTML review pages
# %% ---------------------------------------------------------------------------

# HF token: write access to AlphJain/home-decor-siglip, read access to the data and STL repos.
# Set it as an environment variable before running; never write it into this file.
import os
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN") or ""   # e.g. "hf_..."
assert os.environ["HF_TOKEN"], "Set the HF_TOKEN environment variable"

# %% ---------------------------------------------------------------------------

# 1. Settings
import base64, json
from datetime import datetime, timezone
from pathlib import Path

import torch

EXPERIMENT = "clean"       # "baseline", "real_photos" or "clean" (train split from build_clean_dataset)
DRY_RUN = True             # True = CPU plumbing check (no training, any cluster); then False on a GPU
SMOKE = False              # True = 40-step GPU check on small subsets
# Open Images real photos: an experiment that did not beat the final model (see RESULTS.md); its data scripts
# are not in this repo, so keep USE_OPENIMAGES = False.
USE_OPENIMAGES = False   # True = the v3-clean-oi experiment; False = the final model (v3-clean-nostyle)
OI_RUN = "20261109T110845Z"
RUN_NAME = "v3-clean-oi" if USE_OPENIMAGES else "v3-clean-nostyle"   # clean: DF style off, selection without STL, local checkpoints

DATA_ROOT = "decore"
SOURCE_RUNS = {"home-decore": "20261104T130155Z", "pix3d": "20261104T151426Z", "coco_scenes": "20261104T151655Z"}
CLEAN_RUN = ""   # decore/clean/home-decore/<run> from the latest build_clean_dataset run (set before running)
assert CLEAN_RUN or EXPERIMENT != "clean", "Set CLEAN_RUN to the run id printed by build_clean_dataset"
VAL_RECAPTION = f"{DATA_ROOT}/checks/recaption/val_v1"   # validation text for checkpoint selection (recaption.py)
# ABO validation main photos to skip in the search test (openai_label_jobs Part 2a, ABO_SPLITS=["validation",...]):
# upholstered items whose main photo is only a fabric swatch, and placeholders/drawings. Run id of that run.
ABO_TAG_RUN_HELDOUT = ""
assert ABO_TAG_RUN_HELDOUT or EXPERIMENT != "clean", "Set ABO_TAG_RUN_HELDOUT (openai_label_jobs Part 2a, validation/test)"
MODELS_DIR = f"{DATA_ROOT}/models"                      # checkpoints: <MODELS_DIR>/<run>/checkpoints/step-*/model
LEAK_DIR = f"{DATA_ROOT}/checks/leak/20261105T055610Z"
EXCLUDE_FILE = "exclude_v2.txt"
DATA_REPO = "AlphJain/home-decore"   # only its column schema is read, to rebuild the same columns/types
EXISTING = ["abo", "deepfurniture_crops", "deepfurniture_products", "deepfurniture_scenes"]
OI_SOURCES = ["openimages_crops", "openimages_scenes"]
EXTRA = (["pix3d", "coco_scenes"] if EXPERIMENT == "real_photos"
         else OI_SOURCES if EXPERIMENT == "clean" and USE_OPENIMAGES else [])
# local scratch: <config>/<split>/ Arrow folders (own folder for "clean", so old uncleaned folders are never reused)
LOCAL_DATA = Path(f"/tmp/decore/train_data_clean_{CLEAN_RUN}_valrc{'_oi' + OI_RUN if EXTRA == OI_SOURCES else ''}"
                  if EXPERIMENT == "clean" else "/tmp/decore/train_data")

# Run B's recipe. Both experiments use the same step count so they can be compared.
SETTINGS = {
    "HOME_DECOR_RUN_NAME": RUN_NAME if EXPERIMENT == "clean" else f"v3-{EXPERIMENT}",
    "HOME_DECOR_EXTRA": ",".join(EXTRA),
    "HOME_DECOR_LOCAL_DATA": str(LOCAL_DATA),
    "HOME_DECOR_WORK": "/tmp/decore/train",
    "HOME_DECOR_SMOKE": "1" if SMOKE else "0",
    "HOME_DECOR_STEPS": "1476",          # Run B's planned length (3 epochs at batch 128)
    "HOME_DECOR_EFF_BATCH": "128",       # Run B batch
    "HOME_DECOR_CHUNK": "64",            # images on the GPU at once; lower to 32 if out of memory
    "HOME_DECOR_LR": "5e-6",
    "HOME_DECOR_DISTILL_WEIGHT": "0.1",
    "HOME_DECOR_PAIR_LOSS": "infonce",   # the change that made Run B work
    "HOME_DECOR_PAIR_WEIGHT": "1.0",
    "HOME_DECOR_AUG": "0",               # Run B had no augmentation
    "HOME_DECOR_FIELD_SAMPLE_K": "99",   # all text fields every step, as in Run B
    "HOME_DECOR_HARD_NEG": "0",
    "HOME_DECOR_EVAL_EVERY": "250",
    "HOME_DECOR_PUBLIC_REPO": "",        # never publish publicly from here
    # clean data has re-captions for scenes, so the repeated template titles are dropped
    "HOME_DECOR_SCENE_TITLE": "drop" if EXPERIMENT == "clean" else "keep",
    # clean experiment: DeepFurniture style text off (captions name the style), selection on the validation split
    # over several data types (no STL), weights to the data folder (bf16, best + last)
    "HOME_DECOR_DF_STYLE": "drop" if EXPERIMENT == "clean" else "keep",
    "HOME_DECOR_SELECTION": "multi" if EXPERIMENT == "clean" else "stl",
    "HOME_DECOR_STORE": "local" if EXPERIMENT == "clean" else "hf",
    "HOME_DECOR_MODELS_DIR": MODELS_DIR,
    "HOME_DECOR_DRY_RUN": "1" if DRY_RUN else "0",
}
if not DRY_RUN:
    assert torch.cuda.is_available(), "A GPU is required"
print("experiment:", EXPERIMENT, "| dry run:", DRY_RUN, "| smoke:", SMOKE,
      "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

# %% ---------------------------------------------------------------------------

# 2. Build the local training data from the data folder (skips folders that already exist from an earlier run)
import shutil

from datasets import (Dataset, Features, Image as ImageFeature, Sequence, Value, concatenate_datasets,
                      load_dataset_builder, load_from_disk)


def read_text(path):
    return fs.head(path, fs.ls(path)[0].size)


def shard_paths(folder):
    return sorted(f.path for f in fs.ls(folder) if f.name.endswith(".jsonl"))


def download_shards(folder, local_dir):
    # Copy the stored shards to local disk first (plain files, then datasets are built from them).
    local_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for path in shard_paths(folder):
        local = local_dir / path.rstrip("/").split("/")[-1]
        if not local.exists():
            local.with_suffix(".part").write_text(read_text(path))
            local.with_suffix(".part").rename(local)
        files.append(str(local))
    return files


def rows(paths, key, skip, drop, extra=None):
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line:
                continue
            r = json.loads(line)
            if key(r) in skip:
                continue
            image = base64.b64decode(r.pop("image_b64"))
            if extra:
                r |= extra(r)
            for column in drop:
                r.pop(column, None)
            yield r | {"image": {"bytes": image, "path": None}}


# Validation text for checkpoint selection: the queries/captions written for the validation images
val_recap = {}
if EXPERIMENT == "clean":
    for config in ("abo", "deepfurniture_scenes", "deepfurniture_crops"):
        for path in shard_paths(f"{VAL_RECAPTION}/{config}"):
            for line in read_text(path).splitlines():
                if line:
                    rec = json.loads(line); val_recap[rec["key"]] = rec
    print(f"validation re-captions: {len(val_recap):,}")
    val_abo_skip = set(read_text(f"{DATA_ROOT}/checks/clean/{ABO_TAG_RUN_HELDOUT}/abo_validation/exclude.txt").split())
    print(f"ABO validation main photos skipped in the search test: {len(val_abo_skip)}")


def val_text(config):
    def extra(r):
        key = f"abo/group:{r['group_id']}" if config == "abo" else f"{config}/validation:{r['row_index']}"
        rec = val_recap.get(key, {})
        wrong = bool(rec.get("rc_wrong_item"))   # crops that are not their category: no queries
        if config == "abo" and f"abo/validation:{r['row_index']}" in val_abo_skip:
            wrong = True                          # swatch / placeholder main photo: no queries
        return {"rc_caption": "" if wrong else rec.get("rc_caption", ""),
                "rc_queries": [] if wrong else rec.get("rc_queries", []), "rc_room": rec.get("rc_room", "")}
    return extra


EXTRA_FEATURES = {
    "pix3d": Features({"image_id": Value("string"), "group_id": Value("string"), "category": Value("string"),
                       "category_coarse": Value("string"), "title": Value("string"), "source_image": Value("string"),
                       "bbox": Sequence(Value("int32")), "license": Value("string"), "attribution": Value("string"),
                       "image": ImageFeature()}),
    "coco_scenes": Features({"image_id": Value("string"), "group_id": Value("string"), "title": Value("string"),
                             "captions": Sequence(Value("string")), "category": Value("string"),
                             "category_coarse": Value("string"), "license": Value("string"),
                             "license_url": Value("string"), "attribution": Value("string"),
                             "source_url": Value("string"), "image": ImageFeature()}),
}

exclude = {s: set(read_text(f"{LEAK_DIR}/{s}/{EXCLUDE_FILE}").split())
           for s in ["home-decore"] + [s for s in EXTRA if s not in OI_SOURCES]}
print({s: len(x) for s, x in exclude.items()}, "rows excluded by the leak check")

# Open Images: OpenAI text (oi_v1) attached here; rows dropped = leak check + OpenAI flags (not decor / wrong item /
# no caption) + score check. 2% of photos (and their crops) are held out as a small
# validation split (only used by the data audit; checkpoints are picked on the home-decore validation split).
OI_FEATURES = {
    "openimages_crops": Features({"image_id": Value("string"), "group_id": Value("string"), "category": Value("string"),
                                 "category_coarse": Value("string"), "license": Value("string"),
                                 "attribution": Value("string"), "rc_caption": Value("string"),
                                 "rc_queries": Sequence(Value("string")), "image": ImageFeature()}),
    "openimages_scenes": Features({"image_id": Value("string"), "group_id": Value("string"), "license": Value("string"),
                                  "attribution": Value("string"), "rc_caption": Value("string"),
                                  "rc_queries": Sequence(Value("string")), "rc_room": Value("string"),
                                  "image": ImageFeature()}),
}
OI_DROP_COLUMNS = {"openimages_crops": ("scene_id", "oi_class", "bbox", "occluded"),
                   "openimages_scenes": ("oi_split", "objects")}
oi_text, oi_skip = {}, {}
if EXTRA == OI_SOURCES:
    import hashlib
    for source in OI_SOURCES:
        recs = {}
        for path in shard_paths(f"{DATA_ROOT}/checks/recaption/oi_v1/{source}"):
            for line in read_text(path).splitlines():
                if line:
                    rec = json.loads(line); recs[rec["key"].split(":", 1)[1]] = rec   # key -> image_id
        bad = {i for i, rec in recs.items() if rec.get("error") or rec.get("rc_not_decor") or rec.get("rc_wrong_item")
               or not rec.get("rc_caption")}
        leak = set(read_text(f"{DATA_ROOT}/checks/leak/openimages/{OI_RUN}/{source}/exclude.txt").split())
        score = set(read_text(f"{DATA_ROOT}/checks/score/openimages/{OI_RUN}/{source}/exclude.txt").split())
        scene_of = (lambda i: i) if source == "openimages_scenes" else (lambda i: i.rsplit("_", 1)[0])
        held = {i for i in recs if int(hashlib.sha256(scene_of(i).encode()).hexdigest(), 16) % 50 == 0}
        dropped = bad | leak | score
        oi_text[source] = {i: {"rc_caption": rec.get("rc_caption", ""), "rc_queries": rec.get("rc_queries", []),
                               **({"rc_room": rec.get("rc_room", "")} if source == "openimages_scenes" else {})}
                           for i, rec in recs.items()}
        oi_skip[source] = {"train": dropped | held, "validation": dropped | (set(recs) - held)}
        print(f"{source}: captioned {len(recs):,} | flagged {len(bad):,} | leak {len(leak):,} | score {len(score):,} | "
              f"train {len(set(recs) - oi_skip[source]['train']):,} | validation {len(set(recs) - oi_skip[source]['validation']):,}")

jobs = []
for config in EXISTING:
    features = load_dataset_builder(DATA_REPO, config, token=os.environ["HF_TOKEN"]).info.features
    for split in ("train", "validation"):
        folder = f"{DATA_ROOT}/sources/home-decore/{SOURCE_RUNS['home-decore']}/{config}/{split}"
        skip = exclude["home-decore"] if split == "train" else set()
        split_features, extra = features, None
        if EXPERIMENT == "clean" and split == "train":   # cleaned copy; validation stays the original images
            folder, skip = f"{DATA_ROOT}/clean/home-decore/{CLEAN_RUN}/{config}/train", set()
        if EXPERIMENT == "clean" and config in ("abo", "deepfurniture_scenes", "deepfurniture_crops"):
            # re-caption columns (recaption.py): train from the clean build, validation attached here
            split_features = Features({**features, "rc_caption": Value("string"),
                                       "rc_queries": Sequence(Value("string")), "rc_room": Value("string")})
            extra = val_text(config) if split == "validation" else None
        jobs.append((config, split, folder, lambda r, c=config, s=split: f"{c}/{s}:{r['row_index']}", skip,
                     ("row_index",), split_features, extra))
for source in EXTRA:
    for split in ("train", "validation"):
        if source in OI_SOURCES:   # one folder; the split comes from oi_skip
            folder = f"{DATA_ROOT}/sources/openimages/{OI_RUN}/{source.split('_')[1]}/train"
            jobs.append((source, split, folder, lambda r: r["image_id"], oi_skip[source][split], OI_DROP_COLUMNS[source],
                         OI_FEATURES[source], lambda r, s=source: oi_text[s].get(r["image_id"], {})))
            continue
        folder = f"{DATA_ROOT}/sources/{source}/{SOURCE_RUNS[source]}/{split}"
        jobs.append((source, split, folder, lambda r: r["image_id"], exclude[source], ("source_thumb_b64",),
                     EXTRA_FEATURES[source], None))

for config, split, folder, key, skip, drop, features, extra in jobs:
    out = LOCAL_DATA / config / split
    if out.exists():
        print("exists:", out)
        continue
    paths = download_shards(folder, LOCAL_DATA.parent / "shards" / config / split)
    print(f"{config}/{split}: {len(paths)} shards copied from the data folder", flush=True)
    # One small dataset per shard, saved to disk and memory-mapped back, then joined. All on this machine:
    # no generator, and the whole split is never in memory.
    parts = []
    for n, path in enumerate(paths):
        part_dir = LOCAL_DATA.parent / "build_parts" / config / split / f"{n:05d}"
        if not part_dir.exists():
            part_rows = list(rows([path], key, skip, drop, extra))
            if not part_rows:   # every row of this shard skipped (e.g. the 2% Open Images validation split)
                continue
            Dataset.from_list(part_rows, features=features).save_to_disk(str(part_dir))
        parts.append(load_from_disk(str(part_dir)))
    data = (concatenate_datasets(parts) if parts
            else Dataset.from_dict({name: [] for name in features}, features=features))
    data.save_to_disk(str(out))
    print(f"{config}/{split}: {len(data)} rows -> {out}", flush=True)
    shutil.rmtree(LOCAL_DATA.parent / "build_parts" / config / split, ignore_errors=True)
    shutil.rmtree(LOCAL_DATA.parent / "shards" / config / split, ignore_errors=True)

# %% ---------------------------------------------------------------------------

# 3. Train (settings are read when train_v3 is imported, so they are set first)
import importlib, sys

os.environ.update(SETTINGS)
sys.path.insert(0, HERE)   # train_v3.py is a plain file next to this script
import train_v3
train_v3 = importlib.reload(train_v3)   # picks up changed settings when this cell is run again
train_v3.FS = fs                # local store: checkpoints, pointers and logs are written through fsutil.fs
print("run:", train_v3.RUN_NAME, "| store:", train_v3.STORE, "| selection:", train_v3.SELECTION,
      "| dry run:", train_v3.DRY_RUN, "| sources:", train_v3.SOURCES, "| mix:", train_v3.SOURCE_MIX_OVERRIDE)
train_v3.main()

# %% ---------------------------------------------------------------------------

# 4. Summary per validation (Local runs already keep log.jsonl / config.json in <MODELS_DIR>/<run>/; the old HF runs
#    get a text copy here). Skipped for a dry run, which deletes its outputs.
run_dir = train_v3.RUN_DIR
if not DRY_RUN and (run_dir / "log.jsonl").exists():
    if train_v3.STORE != "local":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_out = f"{DATA_ROOT}/runs/{train_v3.RUN_NAME}/{stamp}"
        for name in ("log.jsonl", "config.json", "data_audit.json"):
            if (run_dir / name).exists():
                assert fs.put(f"{run_out}/{name}", (run_dir / name).read_text(), overwrite=True), name
        print("text copy:", run_out)
    for l in (run_dir / "log.jsonl").read_text().splitlines():
        row = json.loads(l)
        if row.get("event") in ("validation_baseline", "validation"):
            m = row.get("metrics", {})
            sel = m.get("checkpoint_selection", {})
            print(row.get("step"), row.get("event"), "| score", round(sel.get("score", 0), 4) if sel else "-",
                  {k: round(v, 3) for k, v in sel.get("components", {}).items()}, "| best:", row.get("is_best"))
        elif row.get("event") == "done":
            print("done:", row)
