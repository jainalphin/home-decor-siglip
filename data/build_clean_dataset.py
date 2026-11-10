# # Build the clean home-decore TRAIN split (CPU, no GPU, no OpenAI)
#
# Reads the stored copy of home-decore and writes a cleaned **train** split in the same JSONL shard format
# (`row_index` kept). Validation/test are not touched, so scores stay comparable with earlier runs.
#
# | Source | Rule |
# |---|---|
# | all | drop leak-check rows (`exclude_v2.txt`) |
# | ABO | drop photos that ABO lists under a different product (all marketplaces checked) |
# | ABO | drop colour-swatch listings |
# | ABO | drop photos the OpenAI job tagged detail / back / other_item / other (`abo/exclude.txt`), except main photos |
# | ABO | keep 'detail' photos of rugs, wall art, throw pillows (pattern / whole artwork); drop upholstered main photos tagged 'detail' (fabric swatches) |
# | ABO | blank `keywords`, `style`, `details`, seller `color` and `material` (wrong or junk too often) |
# | DF crops | blank `vlm_style` (34% wrong in the audit; DeepFurniture's own `style` is kept) |
# | all | optional extra drop lists (`EXTRA_EXCLUDE`, e.g. from `score_clean_dataset`) |
# | DF crops | drop tiny (< 1% of scene), cut off (< 80% inside image), thin slivers (short/long side < 0.2) |
# | DF crops | keep at most 5 crops per product |
# | DF scenes, DF products | unchanged |
#
# Output: `decore/clean/home-decore/<run>/<config>/train/*.jsonl` + `summary.json` (rows dropped per rule).
# Downloads: ABO listing metadata (16 gz files, public ABO download site over HTTPS).


import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "finetuning")]
from fsutil import fs, show_html   # file helpers and HTML review pages
# %% ---------------------------------------------------------------------------

# 1. Settings
import base64, gzip, hashlib, io, json, re
from collections import Counter, defaultdict
from datetime import datetime, timezone

import requests
from PIL import Image

DATA_ROOT = "decore"
SRC = f"{DATA_ROOT}/sources/home-decore/20261104T130155Z"
LEAK = f"{DATA_ROOT}/checks/leak/20261105T055610Z/home-decore/exclude_v2.txt"
ABO_TAG_RUN = ""   # run id of the FULL OpenAI part 2a run, e.g. "20261108T120000Z" (checks/clean/<run>/abo/exclude.txt)
EXTRA_EXCLUDE = []   # optional extra drop lists, e.g. [f"{DATA_ROOT}/checks/score/<run>/exclude.txt"] from score_clean_dataset
RECAPTION = ""       # optional: f"{DATA_ROOT}/checks/recaption/full_v1" (adds rc_caption/rc_queries/rc_room, drops wrong items)
RC_CONFIGS = ("abo", "deepfurniture_scenes", "deepfurniture_crops")   # ABO: queries only, keyed by product group
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = f"{DATA_ROOT}/clean/home-decore/{RUN_ID}"

# Blanked text fields. Audit 20261108T064534Z (gpt-4.1-mini judge, 200 rows/source): ABO seller color 17% /
# material 12% wrong vs VLM colours 3% / materials 6%; DF crop vlm_style 34% wrong vs DF style 22%.
ABO_BLANK = ("keywords", "style", "details", "color", "material")
CROP_BLANK = ("vlm_style",)
MIN_AREA, MIN_INSIDE, MIN_ASPECT, MAX_PER_PRODUCT = 0.01, 0.8, 0.2, 5
assert ABO_TAG_RUN, "Set ABO_TAG_RUN to the full part 2a run"
print("output:", OUT)


def read_text(path):
    return fs.head(path, fs.ls(path)[0].size)


def shard_paths(folder):
    return sorted(f.path for f in fs.ls(folder) if f.name.endswith(".jsonl"))


def lines(path):
    return (json.loads(l) for l in read_text(path).splitlines() if l)


def put(path, text):
    assert fs.put(path, text, overwrite=True), path


leak = set(read_text(LEAK).split())
abo_tag_drop = set(read_text(f"{DATA_ROOT}/checks/clean/{ABO_TAG_RUN}/abo/exclude.txt").split())
# main photos tagged "other" (placeholder such as "Image Coming Soon", drawing, text graphic) are dropped too
abo_tags = list(lines(f"{DATA_ROOT}/checks/clean/{ABO_TAG_RUN}/abo/tags.jsonl"))
abo_other = {t["key"] for t in abo_tags if t.get("tag") == "other"}
# Per-category photo rules, decided by eye on 2026-11-08 (main vs dropped 'detail' photo sheets):
# - 'detail' photos of rugs / wall art / throw pillows show the pattern or the whole artwork -> keep them
# - an upholstered item whose MAIN photo is tagged 'detail' shows only a fabric swatch -> drop that photo
KEEP_DETAIL = {"rug", "wall art", "throw pillow"}
UPHOLSTERED = {"chair", "sofa", "ottoman", "headboard", "stool", "bean bag", "bench", "bed", "outdoor furniture"}
abo_detail = {t["key"] for t in abo_tags if t.get("tag") == "detail"}
abo_main_detail = {t["key"] for t in abo_tags if t.get("tag") == "detail" and t.get("is_main")}
extra_drop = set().union(*(set(read_text(p).split()) for p in EXTRA_EXCLUDE))
recaptions = {}
if RECAPTION:
    for config in RC_CONFIGS:
        try:
            paths = shard_paths(f"{RECAPTION}/{config}")
        except Exception:
            print(f"no recaptions for {config}"); paths = []
        for path in paths:
            for rec in lines(path):
                recaptions[rec["key"]] = rec
print(f"leak rows: {len(leak):,} | ABO photos tagged bad: {len(abo_tag_drop):,} | extra drops: {len(extra_drop):,}"
      f" | recaptions: {len(recaptions):,}")

# %% ---------------------------------------------------------------------------

# 2. ABO listing metadata: every photo of every item, over all marketplaces
ABO = "https://amazon-berkeley-objects.s3.amazonaws.com/listings/metadata/listings_{}.json.gz"
own_photos = defaultdict(set)
for c in "0123456789abcdef":
    for line in gzip.open(io.BytesIO(requests.get(ABO.format(c), timeout=120).content), "rt"):
        r = json.loads(line)
        own_photos[r["item_id"]].update(i for i in [r.get("main_image_id")] + r.get("other_image_id", []) if i)
print(f"ABO items: {len(own_photos):,}")

# %% ---------------------------------------------------------------------------

# 3. Pass 1 over DF crops (metadata only): decide which crops to keep


def crop_reason(r):
    if f"deepfurniture_crops/train:{r['row_index']}" in leak:
        return "leak"
    if (r.get("area_ratio") or 0) < MIN_AREA:
        return "tiny"
    if (r.get("in_image_ratio") or 0) < MIN_INSIDE:
        return "cut_off"
    w, h = Image.open(io.BytesIO(base64.b64decode(r["image_b64"]))).size
    if min(w, h) / max(w, h) < MIN_ASPECT:
        return "thin_sliver"
    return ""


crop_drop, survivors = {}, defaultdict(list)
for path in shard_paths(f"{SRC}/deepfurniture_crops/train"):
    for r in lines(path):
        reason = crop_reason(r)
        if reason:
            crop_drop[r["row_index"]] = reason
        else:
            survivors[str(r["furniture_id"])].append(r["row_index"])
for pid, rows in survivors.items():   # deterministic pick of MAX_PER_PRODUCT per product
    rows.sort(key=lambda i: hashlib.sha256(f"{pid}:{i}".encode()).hexdigest())
    for i in rows[MAX_PER_PRODUCT:]:
        crop_drop[i] = "over_cap"
print("DF crops dropped:", Counter(crop_drop.values()))

# %% ---------------------------------------------------------------------------

# 4. Pass 2: write the clean train split


def abo_reason(r):
    key = f"abo/train:{r['row_index']}"
    if key in leak:
        return "leak"
    if r["image_id"] not in own_photos.get(r["item_id"], ()):
        return "wrong_photo"
    if "swatch" in f"{r.get('title', '')} {r.get('raw_title', '')}".lower():
        return "swatch"
    category = r.get("category")
    if r.get("is_main") and key in abo_main_detail and category in UPHOLSTERED:
        return "photo_type_main_swatch"               # main photo is a fabric swatch, captioned as the furniture
    if key in abo_detail and category in KEEP_DETAIL:
        return ""                                     # pattern / whole-artwork views: kept (checked by eye)
    if key in abo_tag_drop and not r.get("is_main"):   # main photo kept (so no product disappears) ...
        return "photo_type"
    if key in abo_other:                              # ... unless it is a placeholder / drawing / graphic
        return "photo_type_main_other"
    return ""


def reason_for(config, r):
    if f"{config}/train:{r['row_index']}" in extra_drop:
        return "score_check"
    if recaptions.get(f"{config}/train:{r['row_index']}", {}).get("rc_wrong_item"):
        return "rc_wrong_item"
    if config == "abo":
        return abo_reason(r)
    if config == "deepfurniture_crops":
        return crop_drop.get(r["row_index"], "")
    return "leak" if f"{config}/train:{r['row_index']}" in leak else ""


summary = {}
for config in ("abo", "deepfurniture_crops", "deepfurniture_scenes", "deepfurniture_products"):
    counts, kept = Counter(), 0
    for n, path in enumerate(shard_paths(f"{SRC}/{config}/train")):
        out = []
        for r in lines(path):
            reason = reason_for(config, r)
            counts[reason or "kept"] += 1
            if reason:
                continue
            for field in {"abo": ABO_BLANK, "deepfurniture_crops": CROP_BLANK}.get(config, ()):
                r[field] = ""
            if RECAPTION and config in RC_CONFIGS:   # every row of these configs gets the same three columns
                key = f"abo/group:{r['group_id']}" if config == "abo" else f"{config}/train:{r['row_index']}"
                rec = recaptions.get(key, {})
                r["rc_caption"], r["rc_queries"] = rec.get("rc_caption", ""), rec.get("rc_queries", [])
                r["rc_room"] = rec.get("rc_room", "")
                counts["with_recaption" if (rec.get("rc_caption") or rec.get("rc_queries")) else "no_recaption"] += 1
            out.append(json.dumps(r))
        if out:
            put(f"{OUT}/{config}/train/part-{n:05d}.jsonl", "\n".join(out) + "\n")
        kept += len(out)
    summary[config] = dict(counts)
    print(config, dict(counts), flush=True)

summary["settings"] = {"source": SRC, "leak": LEAK, "abo_tag_run": ABO_TAG_RUN, "extra_exclude": EXTRA_EXCLUDE,
                       "recaption": RECAPTION,
                       "abo_blank": ABO_BLANK, "crop_blank": CROP_BLANK,
                       "min_area": MIN_AREA, "min_inside": MIN_INSIDE, "min_aspect": MIN_ASPECT,
                       "max_per_product": MAX_PER_PRODUCT}
put(f"{OUT}/summary.json", json.dumps(summary, indent=2))
print("saved:", OUT)
