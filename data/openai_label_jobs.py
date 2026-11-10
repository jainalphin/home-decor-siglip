# # Label jobs with a small OpenAI vision model (CPU only, no GPU)
#
# | Part | What | Model calls |
# |---|---|---|
# | 0 | COCO: keep indoor home photos only (caption word rule) | none |
# | 1 | **Fair STL test labels**: for each query crop, judge STL's "correct" product and each model's top 5 as same / similar / different | 1 per query |
# | 2 | **Bad training photos**: tag each ABO view (whole product / close-up / back / other item dominates / other); check Pix3D groups for mismatched photos | 1 per product group |
# | 2c | **DeepFurniture crops** (sample): clear / partly hidden / too small / wrong item | 1 per crop |
# | 3 | **Check the text labels** on ABO, DeepFurniture crop and DeepFurniture scene rows | 1 per row |
#₹
# Image quality: only part 1 (same vs similar product) uses "high" detail; the rest use the cheaper "low".
#
# Every part has a `LIMIT` for a small test run first: check a few answers by eye and the token count, then
# raise it. The model only answers short JSON questions about images; it doesn't write training text.
# Results go to the data folder: exclude lists under `decore/checks/clean/<run>/`, STL labels under `decore/evals/stl_gold/<run>/`.
#
# Part 1 uses the top-10 lists the training runs saved for the 500 STL validation queries
# (`runs/<run>/artifacts/stl_top10_*.jsonl` in the HF model repo), so no model has to run here. Those queries
# were also used to pick checkpoints, a small bias that doesn't affect the base model.


import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "finetuning")]
from fsutil import fs, show_html   # file helpers and HTML review pages
# %% ---------------------------------------------------------------------------

# Keys: set as an environment variable before running; never write it into this file.
# HF token: read access to AlphJain/home-decor-siglip and AlphJain/stl-home-eval (part 1 only).
import os
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY") or ""   # "sk-..."  (if PROVIDER = "openai")
os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY") or ""       # "gsk_..." (if PROVIDER = "groq")
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN") or ""               # "hf_..."
assert os.environ["OPENAI_API_KEY"] or os.environ["GROQ_API_KEY"], "Set OPENAI_API_KEY (or GROQ_API_KEY)"

# %% ---------------------------------------------------------------------------

# 1. Settings and helpers
import base64, io, json, random, re, tarfile, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
from PIL import Image, ImageOps

PROVIDER = "openai"        # "openai" or "groq" (Groq's API is OpenAI-compatible)
MODEL = "gpt-4.1-mini"     # openai: e.g. "gpt-4o-mini", "gpt-4.1-nano"; groq: e.g. "meta-llama/llama-4-scout-17b-16e-instruct"
# Groq allows only a few images per request (3-5 depending on the model; check its model page).
MAX_IMAGES = 18 if PROVIDER == "openai" else 3
PRICE_IN, PRICE_OUT = 0.0, 0.0   # your price per 1M input / output tokens, for the cost estimate (optional)
WORKERS = 8                # parallel requests
# Image quality per job: "high" only where small details decide the answer (part 1: same vs similar product).
# Coarse questions (photo type, same furniture model, colour/material/style) use the much cheaper "low".
SIDE = {"high": 1024, "low": 512}

DATA_ROOT = "decore"
SOURCE_RUNS = {"home-decore": "20261104T130155Z", "pix3d": "20261104T151426Z", "coco_scenes": "20261104T151655Z"}
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
CLEAN_DIR = f"{DATA_ROOT}/checks/clean/{RUN_ID}"
SEED = 0
client = (OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")
          if PROVIDER == "groq" else OpenAI())
usage = Counter()
Image.MAX_IMAGE_PIXELS = None


def read_text_file(path):
    return fs.head(path, fs.ls(path)[0].size)


def read_rows(folder, shards=None):
    paths = sorted(f.path for f in fs.ls(folder) if f.name.endswith(".jsonl"))
    if shards:
        paths = random.Random(SEED).sample(paths, min(shards, len(paths)))
    return [json.loads(l) for p in paths for l in read_text_file(p).splitlines() if l]


def put(path, text):
    assert fs.put(path, text, overwrite=True), path


def small_jpeg_b64(image, side=512):
    # PIL image, raw bytes or base64 string -> base64 JPEG no larger than `side`
    if isinstance(image, str):
        image = base64.b64decode(image)
    if isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image))
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((side, side))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode()


def image_part(image, detail):
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{small_jpeg_b64(image, SIDE[detail])}",
                                               "detail": detail}}


def ask(prompt, images, detail="low", tries=4):
    # One question with images; the answer must be a JSON object. Returns the parsed dict (or {"error": ...}).
    content = [{"type": "text", "text": prompt}] + [image_part(i, detail) for i in images]
    for attempt in range(tries):
        try:
            response = client.chat.completions.create(
                model=MODEL, temperature=0, response_format={"type": "json_object"},
                messages=[{"role": "user", "content": content}])
            usage["input"] += response.usage.prompt_tokens
            usage["output"] += response.usage.completion_tokens
            usage["calls"] += 1
            return json.loads(response.choices[0].message.content)
        except Exception as exc:
            if attempt == tries - 1:
                return {"error": str(exc)[:300]}
            time.sleep(2 ** attempt * 2)


def run_parallel(fn, items):
    with ThreadPoolExecutor(WORKERS) as pool:
        return list(pool.map(fn, items))


def cost_report(label, n):
    cost = (usage["input"] * PRICE_IN + usage["output"] * PRICE_OUT) / 1e6
    print(f"{label}: {usage['calls']} calls so far, {usage['input']:,} input + {usage['output']:,} output tokens"
          + (f", about ${cost:.2f}" if PRICE_IN else "") + (f" | per item: {usage['input'] / max(1, n):,.0f} input" if n else ""))


def thumb(image, side=150):
    return f"<img src='data:image/jpeg;base64,{small_jpeg_b64(image)}' width={side}>"

print("model:", MODEL, "| output:", CLEAN_DIR)

# %% ---------------------------------------------------------------------------

# Part 0. COCO: keep indoor home photos only (no model). A photo is kept if its captions mention a room or
# indoor furniture and don't mention street/park/outdoor words.
ROOM_WORDS = ["living room", "bedroom", "kitchen", "dining room", "bathroom", "office", "room", "couch", "sofa",
              "bed", "fireplace", "bookshelf", "bookcase", "armchair", "coffee table", "dresser", "nightstand",
              "lamp", "rug", "curtain", "interior", "hotel", "apartment", "den", "loft"]
OUTDOOR_WORDS = ["street", "sidewalk", "park", "outdoor", "outside", "road", "beach", "field", "grass", "yard",
                 "patio", "parking", "bus", "train", "airport", "snow", "lawn", "garden", "playground", "boat",
                 "ship", "truck", "car", "bike", "bicycle", "skateboard", "horse", "cow", "dog", "cat", "elephant"]
room_re = re.compile(r"\b(" + "|".join(map(re.escape, ROOM_WORDS)) + r")s?\b")
out_re = re.compile(r"\b(" + "|".join(map(re.escape, OUTDOOR_WORDS)) + r")s?\b")

coco_drop, coco_examples = [], []
for split in ("train", "validation"):
    for r in read_rows(f"{DATA_ROOT}/sources/coco_scenes/{SOURCE_RUNS['coco_scenes']}/{split}"):
        text = " ".join(r.get("captions") or [r.get("title", "")]).lower()
        if out_re.search(text) or not room_re.search(text):
            coco_drop.append(r["image_id"])
            if len(coco_examples) < 30:
                coco_examples.append((r["image_b64"], r["title"]))
print(f"COCO: dropping {len(coco_drop)} photos")
put(f"{CLEAN_DIR}/coco_scenes/exclude.txt", "\n".join(sorted(coco_drop)) + "\n")
show_html("<h3>COCO dropped (first 30)</h3>" + "".join(
    f"<div style='display:inline-block;width:160px;margin:4px;font-size:11px'>{thumb(b)}<br>{t}</div>"
    for b, t in coco_examples))

# %% ---------------------------------------------------------------------------

# Part 1. Fair STL test labels. For each query: the room-photo crop + candidates (STL's "correct" product(s) and
# the top 5 of base, v3-baseline and v3-real_photos), each judged same / similar / different.
from huggingface_hub import hf_hub_download, snapshot_download

LIMIT_STL = 30             # test first; then 500 (all validation queries saved by training)
MODEL_REPO, STL_REPO = "AlphJain/home-decor-siglip", "AlphJain/stl-home-eval"
STL_ROOT = Path("/tmp/decore/stl-home-eval")
token = os.environ["HF_TOKEN"]
assert token, "Part 1 needs the HF token"

if not (STL_ROOT / "scenes").exists():
    snapshot_download(STL_REPO, repo_type="dataset", token=token, local_dir=STL_ROOT,
                      allow_patterns=["queries.jsonl", "gallery.jsonl", "images.tar"])
    with tarfile.open(STL_ROOT / "images.tar") as tar:
        tar.extractall(STL_ROOT, filter="data")
queries = {(q["scene"], tuple(q["bbox"])): q
           for q in (json.loads(x) for x in (STL_ROOT / "queries.jsonl").read_text().splitlines() if x)}


def dump(run, name):
    path = hf_hub_download(MODEL_REPO, f"runs/{run}/artifacts/{name}", token=token)
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x]


tops = {"base": dump("v3-baseline", "stl_top10_base.jsonl"), "v3-baseline": dump("v3-baseline", "stl_top10_best.jsonl"),
        "v3-clean": dump("v3-clean", "stl_top10_best.jsonl")}
n_queries = len(tops["base"])
assert all(len(v) == n_queries for v in tops.values())


def crop_image(q):   # same crop as training/evaluation
    image = ImageOps.exif_transpose(Image.open(STL_ROOT / "scenes" / f"{q['scene']}.jpg")).convert("RGB")
    w, h = image.size
    l, t, r, b = q["bbox"]
    px, py = (r - l) * .1, (b - t) * .1
    return image.crop((max(0, (l - px) * w), max(0, (t - py) * h), min(w, (r + px) * w), min(h, (b + py) * h)))


def product_image(product):
    return Image.open(STL_ROOT / "products" / f"{product}.jpg")


STL_PROMPT = """Image 1 is a crop of a real room photo. The item of interest is a {category}.
Images 2 to {last} are product photos. For each product photo, compare it with the {category} in image 1:
- "same": the same product (same design, shape, details and colour; ignore lighting and angle)
- "similar": not the same product, but a shopper looking for the item in image 1 would accept it
  (same type, very similar shape, colour and material), or the same design in a different colour
- "different": anything else
Return JSON only: {{"labels": [one of "same", "similar", "different" for each product photo, in order]}}"""


def judge_query(i):
    key = (tops["base"][i]["scene"], tuple(tops["base"][i]["bbox"]))
    q = queries[key]
    candidates = list(dict.fromkeys(list(tops["base"][i]["truth"])
                                    + [p for m in tops for p in tops[m][i]["predicted"][:5]]))
    query_image, labels = crop_image(q), []
    # the crop goes in every call; at most 10 products per call (with up to 17 the model often returned the wrong
    # number of labels), and up to 2 retries when the count is still wrong
    step = min(10, MAX_IMAGES - 1)
    for start in range(0, len(candidates), step):
        part = candidates[start:start + step]
        for attempt in range(3):
            answer = ask(STL_PROMPT.format(category=q["category"], last=len(part) + 1),
                         [query_image] + [product_image(p) for p in part], detail="high")
            got = answer.get("labels", [])
            if len(got) == len(part):
                break
        else:
            return {"query": i, "error": answer.get("error", f"got {len(got)} labels for {len(part)}")}
        labels += got
    return {"query": i, "scene": q["scene"], "bbox": q["bbox"], "category": q["category"],
            "truth": tops["base"][i]["truth"], "labels": dict(zip(candidates, labels))}


picked = sorted(random.Random(SEED).sample(range(n_queries), min(LIMIT_STL, n_queries)))
judged = run_parallel(judge_query, picked)
ok = [j for j in judged if "labels" in j]
print(f"{len(ok)} of {len(judged)} queries judged")
cost_report("part 1", len(ok))

# Scores on the judged queries: exact (= "same") and acceptable (= "same" or "similar") at rank 1 and in top 5
scores = {}
for m in tops:
    hits = defaultdict(list)
    for j in ok:
        labels = [j["labels"].get(p, "different") for p in tops[m][j["query"]]["predicted"][:5]]
        hits["exact@1"].append(labels[0] == "same")
        hits["acceptable@1"].append(labels[0] in ("same", "similar"))
        hits["exact@5"].append("same" in labels)
        hits["acceptable@5"].append(any(l in ("same", "similar") for l in labels))
        hits["stl_label@1 (old metric)"].append(tops[m][j["query"]]["predicted"][0] in j["truth"])
    scores[m] = {k: round(sum(v) / len(v), 3) for k, v in hits.items()}
truth_labels = Counter(j["labels"][p] for j in ok for p in j["truth"] if p in j["labels"])
summary = {"model": MODEL, "queries": len(ok), "scores": scores,
           "how_the_judge_rated_stl_truth": dict(truth_labels)}
print(json.dumps(summary, indent=2))
GOLD_DIR = f"{DATA_ROOT}/evals/stl_gold/{RUN_ID}"
put(f"{GOLD_DIR}/labels.jsonl", "\n".join(json.dumps(j) for j in judged) + "\n")
put(f"{GOLD_DIR}/summary.json", json.dumps(summary, indent=2))

# Spot-check page: compare the judge's labels with your own eye
rows = []
for j in ok[:15]:
    q = queries[(j["scene"], tuple(j["bbox"]))]
    cells = "".join(f"<td>{thumb(product_image(p), 120)}<br>{lab}{' (STL)' if p in j['truth'] else ''}</td>"
                    for p, lab in j["labels"].items())
    rows.append(f"<tr><td>{j['category']}<br>{thumb(crop_image(q), 150)}</td>{cells}</tr>")
show_html(f"<h3>Judge labels (first 15): do you agree?</h3><table border=1 cellpadding=2>{''.join(rows)}</table>")

# %% ---------------------------------------------------------------------------

# Part 2a. ABO photos: tag every image of a product listing.
# ABO_SPLITS = ["train"]: all photos of every train listing (done: run 20261108T053249Z, don't re-run).
# ABO_SPLITS = ["validation", "test"]: MAIN photos only, so the held-out search tests can skip the same bad
#   products as train (an upholstered item whose main photo is only a fabric swatch, or a placeholder/drawing).
ABO_SPLITS = ["validation", "test"]
LIMIT_ABO_GROUPS = None    # train: test with 30 first; validation/test: None (about 533 main photos, cents)
MAX_IMAGES_PER_CALL = min(8, MAX_IMAGES)
DROP_TAGS = {"detail", "back", "other_item", "other"}
UPHOLSTERED = {"chair", "sofa", "ottoman", "headboard", "stool", "bean bag", "bench", "bed", "outdoor furniture"}

ABO_PROMPT = """These {n} images come from one shop listing for: "{title}" (category: {category}).
Tag each image with exactly one of:
- "product": the whole product is clearly visible (any background, including a room)
- "detail": a close-up of part of the product or its texture/fabric
- "back": back side, underside, label, packaging or hardware
- "other_item": a room or scene where a different item is clearly the main subject
- "other": a dimension drawing, text graphic, colour swatch or anything unrelated
Return JSON only: {{"tags": [one tag per image, in order]}}"""


def run_abo_split(split):
    abo = read_rows(f"{DATA_ROOT}/sources/home-decore/{SOURCE_RUNS['home-decore']}/abo/{split}")
    if split != "train":
        abo = [r for r in abo if r.get("is_main")]
    groups = defaultdict(list)
    for r in abo:
        groups[r["group_id"]].append(r)
    group_ids = sorted(groups)
    if LIMIT_ABO_GROUPS:
        group_ids = random.Random(SEED).sample(group_ids, min(LIMIT_ABO_GROUPS, len(group_ids)))
    jobs = []
    for g in group_ids:
        rows_ = sorted(groups[g], key=lambda r: not r.get("is_main"))   # main image first
        for start in range(0, len(rows_), MAX_IMAGES_PER_CALL):
            jobs.append(rows_[start:start + MAX_IMAGES_PER_CALL])

    def tag_abo(chunk):
        answer = ask(ABO_PROMPT.format(n=len(chunk), title=chunk[0].get("title", ""),
                                       category=chunk[0].get("category", "")), [r["image_b64"] for r in chunk])
        tags = answer.get("tags", [])
        if len(tags) != len(chunk):
            return [{"key": f"abo/{split}:{r['row_index']}", "tag": "error", "is_main": bool(r.get("is_main")),
                     "category": r.get("category"), "group_id": r["group_id"]} for r in chunk]
        return [{"key": f"abo/{split}:{r['row_index']}", "tag": tag, "is_main": bool(r.get("is_main")),
                 "category": r.get("category"), "group_id": r["group_id"]} for r, tag in zip(chunk, tags)]

    abo_tags = [t for chunk in run_parallel(tag_abo, jobs) for t in chunk]
    print(split, Counter(t["tag"] for t in abo_tags))
    cost_report(f"part 2a {split}", len(abo_tags))
    folder = f"{CLEAN_DIR}/abo" if split == "train" else f"{CLEAN_DIR}/abo_{split}"
    put(f"{folder}/tags.jsonl", "\n".join(json.dumps(t) for t in abo_tags) + "\n")
    if split == "train":
        drop = sorted(t["key"] for t in abo_tags if t["tag"] in DROP_TAGS)
    else:   # the train rule for main photos: upholstered + 'detail' (fabric swatch), or 'other' (placeholder)
        drop = sorted(t["key"] for t in abo_tags
                      if (t["tag"] == "detail" and t.get("category") in UPHOLSTERED) or t["tag"] == "other")
    put(f"{folder}/exclude.txt", "\n".join(drop) + "\n")
    print(f"{split}: {len(drop)} photos to skip -> {folder}/exclude.txt")
    by_key = {f"abo/{split}:{r['row_index']}": r for r in abo}
    shown = [t for t in abo_tags if t["key"] in set(drop)][:60] if split != "train" else abo_tags[:60]
    show_html(f"<h3>ABO {split}: {'photos that will be skipped' if split != 'train' else 'tags (sample)'}"
                f" - do you agree?</h3>" + "".join(
        f"<div style='display:inline-block;width:160px;margin:4px;font-size:12px'>{thumb(by_key[t['key']]['image_b64'])}"
        f"<br><b>{t['tag']}</b> {t.get('category', '')}</div>" for t in shown))


for split in ABO_SPLITS:
    run_abo_split(split)

# %% ---------------------------------------------------------------------------

# Part 2b. Pix3D: in each 3D-model group, flag photos that show a different piece of furniture.
LIMIT_PIX3D_GROUPS = 20    # test first; then None for all (~370 groups)
PIX3D_PROMPT = """These {n} photos are supposed to show the same furniture model ({title}).
List the photos that show a clearly different piece of furniture than the majority (different shape or type;
colour differences alone don't count). Return JSON only: {{"different": [1-based photo numbers, or empty]}}"""

pix3d = [r for split in ("train", "validation")
         for r in read_rows(f"{DATA_ROOT}/sources/pix3d/{SOURCE_RUNS['pix3d']}/{split}")]
pgroups = defaultdict(list)
for r in pix3d:
    pgroups[r["group_id"]].append(r)
pids = [g for g in sorted(pgroups) if len(pgroups[g]) > 1]
if LIMIT_PIX3D_GROUPS:
    pids = random.Random(SEED).sample(pids, min(LIMIT_PIX3D_GROUPS, len(pids)))


def check_pix3d(g):
    rows_ = pgroups[g][:MAX_IMAGES_PER_CALL]
    answer = ask(PIX3D_PROMPT.format(n=len(rows_), title=rows_[0]["title"]), [r["image_b64"] for r in rows_])
    bad = [rows_[k - 1]["image_id"] for k in answer.get("different", []) if isinstance(k, int) and 1 <= k <= len(rows_)]
    return {"group": g, "different": bad, "error": answer.get("error")}


pix3d_checks = run_parallel(check_pix3d, pids)
pix3d_drop = sorted(i for c in pix3d_checks for i in c["different"])
print(f"Pix3D: {len(pix3d_drop)} photos flagged in {len(pids)} groups")
cost_report("part 2b", len(pids))
put(f"{CLEAN_DIR}/pix3d/exclude.txt", "\n".join(pix3d_drop) + "\n")
put(f"{CLEAN_DIR}/pix3d/checks.jsonl", "\n".join(json.dumps(c) for c in pix3d_checks) + "\n")

# %% ---------------------------------------------------------------------------

# Part 2c. DeepFurniture crops (97k, the biggest source): how many are bad? A sample first, to decide whether a
# full pass is worth it. Bad = the item is mostly hidden, too small/blurry, or the crop shows a different item.
LIMIT_DF_CROPS = 50        # test first; then 300 for a solid estimate
DF_PROMPT = """This image is a crop from a rendered room, meant to show one {category}.
Tag it with exactly one of:
- "clear": the {category} is clearly visible and is the main subject
- "partly_hidden": a large part of it is covered by other objects or cut off by the edge
- "too_small": too small, blurry or dark to see its design
- "wrong_item": the main subject is not a {category}
Return JSON only: {{"tag": "..."}}"""

df_crops = read_rows(f"{DATA_ROOT}/sources/home-decore/{SOURCE_RUNS['home-decore']}/deepfurniture_crops/train", shards=3)
df_sample = random.Random(SEED).sample(df_crops, min(LIMIT_DF_CROPS, len(df_crops)))


def tag_df(r):
    answer = ask(DF_PROMPT.format(category=r.get("category", "furniture item")), [r["image_b64"]])
    return {"key": f"deepfurniture_crops/train:{r['row_index']}", "category": r.get("category"),
            "tag": answer.get("tag", "error")}


df_tags = run_parallel(tag_df, df_sample)
df_counts = Counter(t["tag"] for t in df_tags)
print(df_counts, "| bad share:", round(1 - df_counts["clear"] / max(1, len(df_tags)), 3))
cost_report("part 2c", len(df_tags))
put(f"{CLEAN_DIR}/deepfurniture_crops/sample_tags.jsonl", "\n".join(json.dumps(t) for t in df_tags) + "\n")
df_by_key = {f"deepfurniture_crops/train:{r['row_index']}": r for r in df_sample}
show_html("<h3>DeepFurniture crop tags (sample): do you agree?</h3>" + "".join(
    f"<div style='display:inline-block;width:160px;margin:4px;font-size:12px'>{thumb(df_by_key[t['key']]['image_b64'])}"
    f"<br>{t['category']}: <b>{t['tag']}</b></div>" for t in df_tags[:60]))

# %% ---------------------------------------------------------------------------

# Part 3. Are the existing VLM text labels right? 100 ABO + 100 DeepFurniture-crop rows, each field judged.
LIMIT_ROWS = 21            # test first; then 300 (100 ABO, 100 DF crops, 100 DF scenes)
FIELDS = ["vlm_caption", "vlm_style", "vlm_materials", "vlm_colors", "vlm_details",
          "title", "style", "material", "color", "keywords"]
CHECK_PROMPT = """Here is an image of a {category} and labels describing it:
{fields}
For each label, judge it as a description of the item in the image:
"correct", "partly" (some right, some wrong or vague) or "wrong". Use "partly" for anything you can't see.
Return JSON only: {{"<label name>": "correct" | "partly" | "wrong", ...}} with every label name above."""

dfc = read_rows(f"{DATA_ROOT}/sources/home-decore/{SOURCE_RUNS['home-decore']}/deepfurniture_crops/train", shards=2)
dfs = read_rows(f"{DATA_ROOT}/sources/home-decore/{SOURCE_RUNS['home-decore']}/deepfurniture_scenes/train", shards=2)
for r in abo: r["_source"] = "abo"
for r in dfc: r["_source"] = "deepfurniture_crops"
for r in dfs: r["_source"] = "deepfurniture_scenes"; r.setdefault("category", "room")
third = LIMIT_ROWS // 3
sample = (random.Random(SEED).sample(abo, third) + random.Random(SEED).sample(dfc, third)
          + random.Random(SEED).sample(dfs, LIMIT_ROWS - 2 * third))


def check_row(r):
    fields = {f: r[f] for f in FIELDS if r.get(f) not in (None, "", [], "none")}
    answer = ask(CHECK_PROMPT.format(category=r.get("category", "item"),
                                     fields="\n".join(f"- {k}: {v}" for k, v in fields.items())), [r["image_b64"]])
    return {"source": r["_source"], "fields": fields, "verdict": {k: answer.get(k) for k in fields},
            "error": answer.get("error")}


checks = run_parallel(check_row, sample)
cost_report("part 3", len(checks))
table = defaultdict(Counter)
for c in checks:
    for field, verdict in c["verdict"].items():
        table[f"{c['source']}.{field}"][verdict] += 1
field_summary = {f: {**dict(v), "correct_share": round(v["correct"] / max(1, sum(v.values())), 2)}
                 for f, v in sorted(table.items())}
print(json.dumps(field_summary, indent=2))
put(f"{CLEAN_DIR}/label_check/checks.jsonl", "\n".join(json.dumps(c) for c in checks) + "\n")
put(f"{CLEAN_DIR}/label_check/summary.json", json.dumps(field_summary, indent=2))
print("saved:", CLEAN_DIR)
