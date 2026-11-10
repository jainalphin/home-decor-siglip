# # Re-caption DeepFurniture scenes and crops (CPU, OpenAI)
#
# New text written from each image itself, as used for FashionCLIP / Marqo-FashionSigLIP-style training:
#
# | Source | Writes |
# |---|---|
# | DF scenes | `rc_caption` (room type, style, key furniture, colours, materials, light), `rc_queries` (3 search queries), `rc_room` |
# | DF crops | `rc_caption` (the item as it looks in the crop), `rc_queries` (3 search queries), `rc_wrong_item` |
#
# The known metadata (style, object list, category) is given only as a hint; the model describes what it sees.
#
# **Run `LIMIT = 40` first** (pilot): check the old vs new text page and the cost line, then set `LIMIT = None`.
# The full run writes one JSONL per input shard and skips shards already written, so it can be re-run after a
# crash. Output: `decore/checks/recaption/<RUN_NAME>/<config>/part-*.jsonl`, keyed by `<config>/train:<row_index>`.


import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "finetuning")]
from fsutil import fs, show_html   # file helpers and HTML review pages
# %% ---------------------------------------------------------------------------

# Key: set as an environment variable before running; never write it into this file.
import os
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY") or ""   # "sk-..."
assert os.environ["OPENAI_API_KEY"], "Set the OPENAI_API_KEY environment variable"

# %% ---------------------------------------------------------------------------

# 1. Settings and helpers
import base64, io, json, random, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from PIL import Image, ImageOps

DATA_ROOT = "decore"
CLEAN = f"{DATA_ROOT}/clean/home-decore/20261108T070632Z"
MODEL = "gpt-4.1-mini"
PRICE_IN, PRICE_OUT = 0.0, 0.0   # your $ per 1M input / output tokens (for the cost line)
LIMIT = 40                       # pilot: rows per source; None = everything (full run)
SPLIT = "train"                  # "train" (clean build), "validation" or "test" (held-out text, original data)
DATA = CLEAN if SPLIT == "train" else f"{DATA_ROOT}/sources/home-decore/20261104T130155Z"
_short = {"train": "", "validation": "val_", "test": "test_"}[SPLIT]   # validation keeps its existing folder name
RUN_NAME = f"{_short}pilot" if LIMIT else ("full_v1" if SPLIT == "train" else f"{_short}v1")
OUT = f"{DATA_ROOT}/checks/recaption/{RUN_NAME}"
SOURCES = ["deepfurniture_scenes", "deepfurniture_crops"]   # or ["abo"]: 3 search queries per product (main photo)
WORKERS, SEED = 16, 0
client, usage = OpenAI(), Counter()


def read_text(path):
    return fs.head(path, fs.ls(path)[0].size)


def exists(path):
    try:
        fs.ls(path); return True
    except Exception:
        return False


def shard_paths(folder):
    return sorted(f.path for f in fs.ls(folder) if f.name.endswith(".jsonl"))


def put(path, text):
    assert fs.put(path, text, overwrite=True), path


def jpeg_b64(b64, side=512):
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(base64.b64decode(b64)))).convert("RGB")
    image.thumbnail((side, side)); buf = io.BytesIO(); image.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def ask(prompt, image_b64, tries=5):
    content = [{"type": "text", "text": prompt},
               {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64(image_b64)}",
                                                   "detail": "low"}}]
    for attempt in range(tries):
        try:
            r = client.chat.completions.create(model=MODEL, temperature=0.3, response_format={"type": "json_object"},
                                               messages=[{"role": "user", "content": content}])
            usage["input"] += r.usage.prompt_tokens; usage["output"] += r.usage.completion_tokens; usage["calls"] += 1
            return json.loads(r.choices[0].message.content)
        except Exception as exc:
            if attempt == tries - 1:
                return {"error": str(exc)[:300]}
            time.sleep(2 ** attempt * 2)


def cost_line(n):
    cost = (usage["input"] * PRICE_IN + usage["output"] * PRICE_OUT) / 1e6
    return (f"{usage['calls']} calls, {usage['input']:,} in + {usage['output']:,} out tokens"
            + (f", about ${cost:.2f}" if PRICE_IN else "")
            + (f" | per item {usage['input'] / max(1, n):,.0f} in, {usage['output'] / max(1, n):,.0f} out" if n else ""))

print("model:", MODEL, "| split:", SPLIT, "| data:", DATA, "| output:", OUT)

# %% ---------------------------------------------------------------------------

# 2. Prompts
SCENE_PROMPT = """You are writing training text for an interior-design image search engine.
Hint from the dataset (may be incomplete or partly wrong): style "{style}"; objects: {objects}.
Describe ONLY what is visible in the image. No brand names. Do not say it is a render, 3D image or photo.
Return JSON only:
{{"room": one of "living room", "bedroom", "dining room", "kitchen", "bathroom", "study", "hallway", "kids room", "other",
 "caption": one sentence of 15-30 words as an interior-design magazine would write it: room type, style, the main
   furniture pieces, key colours and materials, light. Concrete details only: no closing phrase about
   atmosphere or mood (avoid "creating a calm/cozy/inviting atmosphere"),
 "queries": [3 different short searches (2-6 words) someone might type to find this image: one about the style,
   one about a key piece, one about colour or material. Each must name something specific; avoid filler words
   like "design", "decor", "ideas"]}}"""

CROP_PROMPT = """You are writing training text for an interior-design product search engine.
The image is a crop of one {category} from a room picture (style hint, may be wrong: "{style}").
Describe the {category} itself, not the room. Describe ONLY what is visible. No brand names.
Return JSON only:
{{"wrong_item": true if the main subject is clearly not a {category}, else false,
 "caption": one sentence of 10-20 words: item type, shape, colour, material, notable details, style
   (empty string if wrong_item),
 "queries": [3 different short searches (2-6 words) a shopper might type to find this item; each must name
   something specific (type, colour, material or shape); avoid filler words like "design", "decor"]}}"""


ABO_PROMPT = """You are writing training text for an interior-design product search engine.
The image is a shop photo of one {category}. Shop title: "{title}".
Return JSON only:
{{"queries": [3 different short searches (2-6 words) a shopper might type to find this exact item, based on what is
   visible: e.g. one about the item type and style, one about colour or material, one about a notable detail.
   Each must name something specific; no brand names, sizes or prices; avoid filler words like "design", "decor"]}}"""


def recaption(r, source):
    key = f"{source}/{SPLIT}:{r['row_index']}"
    if source == "abo":   # queries only; ABO captions are already good. One call per product, keyed by group.
        answer = ask(ABO_PROMPT.format(category=r.get("category") or "home item", title=r.get("title") or ""),
                     r["image_b64"])
        out = {"key": f"abo/group:{r['group_id']}"}
    elif source == "deepfurniture_scenes":
        answer = ask(SCENE_PROMPT.format(style=r.get("style") or "unknown",
                                         objects=", ".join(r.get("categories") or []) or "unknown"), r["image_b64"])
        out = {"key": key, "rc_room": answer.get("room", "")}
    else:
        answer = ask(CROP_PROMPT.format(category=r.get("category") or "furniture item", style=r.get("style") or "unknown"),
                     r["image_b64"])
        out = {"key": key, "rc_wrong_item": bool(answer.get("wrong_item", False))}
    queries = answer.get("queries") if isinstance(answer.get("queries"), list) else []
    return out | {"rc_caption": str(answer.get("caption", "")).strip(),
                  "rc_queries": [str(q).strip() for q in queries if str(q).strip()][:3],
                  "error": answer.get("error")}

# %% ---------------------------------------------------------------------------

# 3. Run. Pilot: LIMIT rows per source from random shards. Full: every shard, skipping shards already written.
results = {}
with ThreadPoolExecutor(WORKERS) as pool:
    for source in SOURCES:
        paths = shard_paths(f"{DATA}/{source}/{SPLIT}")
        if LIMIT:
            paths = random.Random(SEED).sample(paths, min(4, len(paths)))
        done_rows = []
        for n, path in enumerate(paths):
            out_path = f"{OUT}/{source}/{path.rstrip('/').split('/')[-1]}"
            if not LIMIT and exists(out_path):
                continue
            rows = [json.loads(l) for l in read_text(path).splitlines() if l]
            if source == "abo":   # main photo only: one call per product
                rows = [r for r in rows if r.get("is_main")]
            if LIMIT:
                rows = random.Random(SEED + n).sample(rows, min(-(-LIMIT // len(paths)), len(rows)))
            got = list(pool.map(lambda r: recaption(r, source), rows))
            put(out_path, "\n".join(json.dumps(g) for g in got) + "\n")
            done_rows += [(r, g) for r, g in zip(rows, got)] if LIMIT else []
            print(f"{source} {n + 1}/{len(paths)}: {len(got)} rows | errors {sum(bool(g['error']) for g in got)}"
                  f" | {cost_line(usage['calls'])}", flush=True)
        results[source] = done_rows
print("done:", cost_line(usage["calls"]))

# %% ---------------------------------------------------------------------------

# 4. Pilot check: old text vs new text. Do the new captions describe the image better? Any invented details?
if LIMIT:
    cards = []
    for source, pairs in results.items():
        cards.append(f"<h3>{source}</h3>")
        for r, g in pairs:
            old = r.get("vlm_caption") if source == "deepfurniture_crops" else r.get("title")
            flag = " <b style='color:#b00'>WRONG ITEM</b>" if g.get("rc_wrong_item") else ""
            cards.append(
                f"<div style='display:inline-block;width:300px;vertical-align:top;margin:4px;font-size:11px;"
                f"border:1px solid #ccc;padding:3px'><img src='data:image/jpeg;base64,{jpeg_b64(r['image_b64'], 280)}'>"
                f"<br><b>old:</b> {old}<br><b>new:</b> {g['rc_caption']}{flag}<br><b>queries:</b> "
                f"{' | '.join(g['rc_queries'])}{'<br><b>room:</b> ' + g['rc_room'] if g.get('rc_room') else ''}</div>")
    show_html("".join(cards))
