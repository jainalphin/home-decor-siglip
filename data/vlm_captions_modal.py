"""Phase A full run: Qwen3-VL-32B captions for all ABO + DeepFurniture products (prompt v4).

Prerequisites: same as the pilot (Modal secret ``huggingface-secret`` with HF_TOKEN,
``vocab.json`` next to this file (data/vocab.json)).

1. Smoke test (1 chunk = 500 products):
     modal run modal_vlm_pilot.py --max-chunks 1
2. Full run (resumable; re-run the same command after any interruption):
     modal run modal_vlm_pilot.py
   Parallelism: MAX_CONTAINERS below (default 3 H100s).

What it does:
  - builds a manifest: one item per ABO product (main image, ORIGINAL size from the ABO image server,
    falling back to the 256px copy) + one per DeepFurniture product render, all splits
  - splits it into chunks of 500, captions chunks in parallel on H100s with vLLM
  - each finished chunk is uploaded to AlphJain/home-decor-vlm/full_v4/chunks/,
    so finished chunks are skipped on re-runs
  - invalid values are removed per field (the rest of the answer is kept); the raw model
    output is stored too
"""

import modal
from pathlib import Path

APP_NAME = "home-decor-vlm-full"
MODEL_ID = "Qwen/Qwen3-VL-32B-Instruct"
DATA_REPO = "AlphJain/home-decore"
OUT_REPO = "AlphJain/home-decor-vlm"
RUN_TAG = "full_v4"
CHUNK_SIZE = 500
MAX_CONTAINERS = 3          # parallel H100s; lower it to reduce cost per hour
MAX_CAPTION_WORDS = 20
VLM_MAX_SIDE = 1024
HERE = Path(__file__).resolve().parent
REMOTE_VOCAB = "/app/vocab.json"
CACHE = "/cache"

base = modal.Image.debian_slim(python_version="3.12").pip_install(
    "numpy>=2.0,<3", "pyarrow>=19", "datasets>=3.5",
    "huggingface_hub>=0.30", "pandas>=2.2", "pillow>=11",
)
cpu_image = base.add_local_file(str(HERE / "vocab.json"), REMOTE_VOCAB)
gpu_image = base.pip_install(
    "vllm>=0.11.0", "numpy>=2.0,<3", "pyarrow>=19",
).add_local_file(str(HERE / "vocab.json"), REMOTE_VOCAB)

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("home-decor-vlm-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")
ENV = {"HF_HOME": f"{CACHE}/huggingface", "HF_HUB_CACHE": f"{CACHE}/huggingface/hub",
       "TOKENIZERS_PARALLELISM": "false", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}

# ---------------------------------------------------------------------- shared helpers


def load_vocab():
    import json
    vocab = json.loads(Path(REMOTE_VOCAB).read_text(encoding="utf-8"))
    norm = lambda xs: [str(x).strip().casefold() for x in xs]
    styles_all = norm(vocab["styles"])
    materials = sorted((set(norm(vocab["materials"])) - {"upholstery"})
                       | {"wood", "fabric", "boucle", "metal", "mesh", "paper"})
    colors = norm(vocab["colors"])
    colors += [c for c in ["orange", "clear", "teal"] if c not in colors]
    generic = ["contemporary", "minimalist", "traditional", "transitional"]
    specific = [s for s in styles_all if s not in generic and s != "modern"]
    return {"specific": specific, "generic": generic, "materials": materials, "colors": colors}


def prompt_for(v, category, title=""):
    import json
    title_line = ""
    if title:
        title_line = (f'\nListing title (a hint about WHICH product to describe): "{title}"\n'
                      "Use the title to identify the product (e.g. a table cover, not the table under it).\n"
                      "Judge colors and materials from the image; if the title names a color, map it to\n"
                      "the closest color in the list below.\n")
    return f"""You label visible attributes of ONE home-decor product for search.
The product is: {category}.
{title_line}
Look ONLY at the {category}. Ignore every other object in the image (other furniture,
props, people, rooms, underlying furniture) and ignore any printed text or slogans in the image.

Return exactly one JSON object, no markdown:
{{"style": [], "materials": [], "colors": [], "shape_details": [], "caption": ""}}

style:
- Choose 1 (at most 2) from this list ONLY if its visual cues are clearly present:
  {json.dumps(v["specific"])}
  Cues: mid-century modern = tapered or splayed legs, organic curves, warm wood;
  art deco = geometric symmetry, brass, fan or stepped shapes, lacquer;
  japandi = low profile, light wood, natural textures, very plain;
  scandinavian = light wood, simple soft shapes, pale colors;
  industrial = black metal, exposed frames, raw wood;
  farmhouse / rustic = distressed or rough wood, X-braces, plank tops.
- IMPORTANT: if you see tapered or splayed legs, button tufting, flared arms, or organic
  curves, the answer is "mid-century modern". Never answer just "modern".
- If none of these fits clearly, use ONE fallback from {json.dumps(v["generic"])},
  or ["none"] if even that is unclear. Never combine "none" with another style.

materials: only from {json.dumps(v["materials"])}.
- Use "wood" unless the grain and color clearly show a species (e.g. walnut, oak).
- Use "metal" unless the specific metal (brass, chrome, iron, steel...) is clear.
- Use "fabric" for upholstery unless a specific fabric (velvet, linen, boucle, leather) is clear.
- Return [] if materials are not visible.

colors: at most 3 from {json.dumps(v["colors"])}, colors of the {category} only,
as they appear in the image (not the background or other objects).

shape_details: short visible details such as "tapered legs", "channel tufting"; [] if none.

caption: 10 to 15 words, must start with the product type, describe only the {category},
no brand, measurements, or quoted text from the image.
"""


def parse_and_clean(text, v):
    import json
    import re
    errors = []
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    s, e = cleaned.find("{"), cleaned.rfind("}")
    if s < 0 or e <= s:
        return None, ["no JSON object"]
    try:
        parsed = json.loads(cleaned[s:e + 1])
    except Exception as err:
        return None, [f"bad JSON: {err}"]

    def lst(key):
        val = parsed.get(key)
        return [str(x).strip().casefold() for x in val if str(x).strip()] if isinstance(val, list) else []

    allowed_style = set(v["specific"]) | set(v["generic"]) | {"none"}
    style = [x for x in lst("style") if x in allowed_style]
    if len(style) != len(lst("style")):
        errors.append(f"dropped styles: {[x for x in lst('style') if x not in allowed_style]}")
    if "none" in style and len(style) > 1:
        style = [x for x in style if x != "none"]
        errors.append("none mixed with a style")
    style = style[:2]
    materials = [x for x in lst("materials") if x in set(v["materials"])]
    if len(materials) != len(lst("materials")):
        errors.append(f"dropped materials: {[x for x in lst('materials') if x not in set(v['materials'])]}")
    colors = [x for x in lst("colors") if x in set(v["colors"])]
    if len(colors) != len(lst("colors")):
        errors.append(f"dropped colors: {[x for x in lst('colors') if x not in set(v['colors'])]}")
    colors = colors[:3]
    caption = str(parsed.get("caption", "")).strip()
    if len(caption.split()) > MAX_CAPTION_WORDS:
        caption = " ".join(caption.split()[:MAX_CAPTION_WORDS])
        errors.append("caption truncated")
    return {"vlm_style": [x for x in style if x != "none"], "vlm_materials": materials,
            "vlm_colors": colors, "vlm_details": lst("shape_details")[:5], "vlm_caption": caption}, errors

# ---------------------------------------------------------------------- 1. manifest (CPU)


@app.function(image=cpu_image, cpu=4, memory=16384, timeout=3600, secrets=[hf_secret],
              volumes={CACHE: cache_volume}, env=ENV)
def build_manifest():
    import json
    import os
    import tempfile
    from datasets import load_dataset
    from huggingface_hub import HfApi

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    api.create_repo(OUT_REPO, repo_type="dataset", private=True, exist_ok=True)
    items = []
    for split in ["train", "validation", "test"]:
        abo = load_dataset(DATA_REPO, "abo", split=split, token=token)
        cols = abo.select_columns(["group_id", "is_main", "image_id", "category", "title"]).to_pandas()
        cols["row"] = range(len(cols))
        cols = cols.sort_values(["group_id", "is_main"], ascending=[True, False]).drop_duplicates("group_id")
        for r in cols.itertuples():
            items.append({"item_id": f"abo:{r.group_id}", "source": "abo", "split": split, "row": int(r.row),
                          "group_id": r.group_id, "image_id": r.image_id, "category": r.category,
                          "title": r.title or ""})
        dfp = load_dataset(DATA_REPO, "deepfurniture_products", split=split, token=token)
        meta = dfp.select_columns(["furniture_id", "category"]).to_pandas()
        for i, r in enumerate(meta.itertuples()):
            items.append({"item_id": f"df:{r.furniture_id}", "source": "deepfurniture", "split": split,
                          "row": i, "furniture_id": str(r.furniture_id), "category": r.category, "title": ""})
    chunks = [items[i:i + CHUNK_SIZE] for i in range(0, len(items), CHUNK_SIZE)]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        for ci, chunk in enumerate(chunks):
            for it in chunk:
                fh.write(json.dumps({**it, "chunk": ci}) + "\n")
        path = fh.name
    api.upload_file(path_or_fileobj=path, path_in_repo=f"{RUN_TAG}/manifest.jsonl",
                    repo_id=OUT_REPO, repo_type="dataset", commit_message="full run manifest")
    print(f"manifest: {len(items)} items, {len(chunks)} chunks "
          f"(abo {sum(i['source'] == 'abo' for i in items)}, "
          f"deepfurniture {sum(i['source'] == 'deepfurniture' for i in items)})")
    return len(chunks)


@app.function(image=cpu_image, cpu=1, memory=2048, timeout=600, secrets=[hf_secret],
              volumes={CACHE: cache_volume}, env=ENV)
def get_run_state():
    """Read resumability state inside Modal, where Hub dependencies are installed."""
    import json
    import os
    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    try:
        files = set(api.list_repo_files(OUT_REPO, repo_type="dataset"))
    except Exception:
        return {"manifest_exists": False, "n_chunks": 0, "done": []}

    manifest_name = f"{RUN_TAG}/manifest.jsonl"
    if manifest_name not in files:
        return {"manifest_exists": False, "n_chunks": 0, "done": []}

    path = hf_hub_download(
        OUT_REPO, manifest_name, repo_type="dataset", token=token,
    )
    chunk_ids = [json.loads(line)["chunk"] for line in Path(path).read_text().splitlines()
                 if line.strip()]
    done = sorted(
        int(filename.rsplit("_", 1)[-1].split(".", 1)[0])
        for filename in files
        if filename.startswith(f"{RUN_TAG}/chunks/chunk_") and filename.endswith(".jsonl")
    )
    return {
        "manifest_exists": True,
        "n_chunks": 1 + max(chunk_ids) if chunk_ids else 0,
        "done": done,
    }

# ---------------------------------------------------------------------- 2. captioning (GPU)


@app.cls(image=gpu_image, gpu="H100", cpu=8, memory=65536, timeout=4 * 3600, max_containers=MAX_CONTAINERS,
         secrets=[hf_secret], volumes={CACHE: cache_volume}, env=ENV)
class Captioner:
    @modal.enter()
    def load(self):
        import os
        import pandas as pd
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        self.token = os.environ["HF_TOKEN"]
        self.vocab = load_vocab()
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.llm = LLM(
            model=MODEL_ID,
            dtype="bfloat16",
            max_model_len=4096,
            gpu_memory_utilization=0.95,
            # This run is image-only. Disabling video avoids Qwen3-VL's large
            # video-profile allocation during vLLM engine initialization.
            limit_mm_per_prompt={"image": 1, "video": 0},
            # Keep the single-H100 memory profile conservative. Requests are
            # still submitted in batches and vLLM schedules them internally.
            max_num_seqs=16,
            # Avoid extra CUDA-graph/torch.compile memory during startup.
            enforce_eager=True,
        )
        self.params = SamplingParams(temperature=0.0, max_tokens=320)
        self.data = {}
        for split in ["train", "validation", "test"]:
            self.data[("abo", split)] = load_dataset(DATA_REPO, "abo", split=split, token=self.token)
            self.data[("deepfurniture", split)] = load_dataset(DATA_REPO, "deepfurniture_products",
                                                              split=split, token=self.token)
        csv_path = Path(CACHE) / "abo_images.csv.gz"
        if not csv_path.exists():
            import urllib.request
            urllib.request.urlretrieve(
                "https://amazon-berkeley-objects.s3.amazonaws.com/images/metadata/images.csv.gz", csv_path)
        self.abo_paths = pd.read_csv(csv_path).set_index("image_id")["path"].to_dict()
        manifest = hf_hub_download(OUT_REPO, f"{RUN_TAG}/manifest.jsonl", repo_type="dataset", token=self.token)
        self.manifest = pd.read_json(manifest, lines=True)

    def _image(self, item):
        import io
        import urllib.request
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        src = "render"
        im = None
        if item["source"] == "abo":
            try:
                url = f"https://amazon-berkeley-objects.s3.amazonaws.com/images/original/{self.abo_paths[item['image_id']]}"
                im = Image.open(io.BytesIO(urllib.request.urlopen(url, timeout=30).read()))
                src = "original"
            except Exception:
                src = "small_256"
        if im is None:
            im = self.data[(item["source"], item["split"])][int(item["row"])]["image"]
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        im.thumbnail((VLM_MAX_SIDE, VLM_MAX_SIDE))
        return im, src

    @modal.method()
    def run_chunk(self, chunk_id: int):
        import json
        import tempfile
        from concurrent.futures import ThreadPoolExecutor
        from huggingface_hub import HfApi

        api = HfApi(token=self.token)
        target = f"{RUN_TAG}/chunks/chunk_{chunk_id:05d}.jsonl"
        if api.file_exists(OUT_REPO, target, repo_type="dataset"):
            return {"chunk": chunk_id, "skipped": True}
        items = self.manifest[self.manifest["chunk"] == chunk_id].to_dict("records")
        with ThreadPoolExecutor(16) as ex:
            loaded = list(ex.map(self._image, items))

        requests = []
        for item, (im, _) in zip(items, loaded):
            text = prompt_for(self.vocab, item["category"], item["title"] if item["source"] == "abo" else "")
            chat = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
            prompt = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            requests.append({"prompt": prompt, "multi_modal_data": {"image": im}})
        outputs = self.llm.generate(requests, self.params)

        rows, n_fail = [], 0
        for item, (_, src), out in zip(items, loaded, outputs):
            raw = out.outputs[0].text.strip()
            fields, errors = parse_and_clean(raw, self.vocab)
            if fields is None:
                n_fail += 1
                fields = {"vlm_style": [], "vlm_materials": [], "vlm_colors": [], "vlm_details": [], "vlm_caption": ""}
            rows.append({"item_id": item["item_id"], "source": item["source"], "split": item["split"],
                         "group_id": item.get("group_id"), "furniture_id": item.get("furniture_id"),
                         "category": item["category"], "image_source": src, **fields,
                         "errors": errors, "raw_output": raw, "model": MODEL_ID, "prompt_version": "v4"})
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            path = fh.name
        api.upload_file(path_or_fileobj=path, path_in_repo=target, repo_id=OUT_REPO,
                        repo_type="dataset", commit_message=f"{RUN_TAG} chunk {chunk_id}")
        return {"chunk": chunk_id, "items": len(rows), "parse_failures": n_fail,
                "abo_original": sum(r["image_source"] == "original" for r in rows if r["source"] == "abo")}

# ---------------------------------------------------------------------- 3. entrypoint


@app.local_entrypoint()
def main(max_chunks: int = 0, rebuild_manifest: bool = False):
    state = get_run_state.remote()
    if rebuild_manifest or not state["manifest_exists"]:
        build_manifest.remote()
        state = get_run_state.remote()

    n_chunks = state["n_chunks"]
    done = set(state["done"])
    todo = [c for c in range(n_chunks) if c not in done]
    if max_chunks:
        todo = todo[:max_chunks]
    print(f"{n_chunks} chunks total, {len(done)} already done, running {len(todo)} "
          f"on up to {MAX_CONTAINERS} H100s")
    totals = {"items": 0, "parse_failures": 0, "errors": 0}
    for res in Captioner().run_chunk.map(todo, return_exceptions=True):
        print(res)
        if isinstance(res, dict) and not res.get("skipped"):
            totals["items"] += res["items"]
            totals["parse_failures"] += res["parse_failures"]
        elif not isinstance(res, dict):
            totals["errors"] += 1
    print("this run:", totals, "-> re-run the same command to retry any failed chunks")
