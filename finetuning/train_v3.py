# v3: SigLIP 2 home-decor fine-tune, run from `run_training.py`.
# Same GradCache trainer as lightning_train_v1b.py (the never-run v2 code), plus:
#   - data comes from local Arrow folders the script builds from the data folder (leak-checked rows already removed)
#   - optional extra sources: pix3d (real photos, "same 3D model" image pairs) and coco_scenes (real rooms)
#   - fixes: checkpoint selection starts at the base model's score (a worse model can't be "best");
#     the run log is continued on resume instead of overwritten; the STL queries used for selection are saved
#     so the final test can leave them out
# All settings come from environment variables the script sets before importing this file.

import base64
import gc
import hashlib
import io
import itertools
import json
import math
import os
import random
import re
import shutil
import tarfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

# ------------------------------------------------------------------ settings

SMOKE = os.environ.get("HOME_DECOR_SMOKE", "0") == "1"
USE_LORA = os.environ.get("HOME_DECOR_USE_LORA", "0") == "1"
RUN_NAME = os.environ.get("HOME_DECOR_RUN_NAME", "v3-baseline") + ("-smoke" if SMOKE else "")

DATA_REPO = "AlphJain/home-decore"
STL_REPO = "AlphJain/stl-home-eval"
OUTPUT_REPO = "AlphJain/home-decor-siglip"
PUBLIC_OUTPUT_REPO = os.environ.get(
    "HOME_DECOR_PUBLIC_REPO", ""
).strip()   # empty = never publish publicly
BASE_MODEL = os.environ.get("HOME_DECOR_BASE_MODEL", "google/siglip2-base-patch16-224")
# Extra training sources, e.g. HOME_DECOR_EXTRA="pix3d,coco_scenes". Empty = the Run B data only.
EXTRA = tuple(x.strip() for x in os.environ.get("HOME_DECOR_EXTRA", "").split(",") if x.strip())
CONFIGS = ("abo", "deepfurniture_crops", "deepfurniture_scenes", "deepfurniture_products") + EXTRA
SOURCES = ("abo", "deepfurniture_crops", "deepfurniture_scenes") + EXTRA   # products are render partners only
LOCAL_DATA = Path(os.environ.get("HOME_DECOR_LOCAL_DATA", "/tmp/decore/train_data"))   # <config>/<split>/
STEPS_OVERRIDE = int(os.environ.get("HOME_DECOR_STEPS", "0"))   # >0: fixed step count (to match another run)

SEED = 20261011

# GradCache: EFF_BATCH is the contrastive batch (number of negatives).
# CHUNK is what actually sits in VRAM at once. Raise CHUNK until just under OOM.
EFF_BATCH = int(os.environ.get("HOME_DECOR_EFF_BATCH", "1024"))
CHUNK = int(os.environ.get("HOME_DECOR_CHUNK", "64"))
GRAD_CKPT = os.environ.get("HOME_DECOR_GRAD_CKPT", "1") == "1"

EPOCHS = float(os.environ.get("HOME_DECOR_EPOCHS", "2"))
LR = float(os.environ.get("HOME_DECOR_LR", "1e-4" if USE_LORA else "2e-5"))
WEIGHT_DECAY = 0.1
WARMUP_FRACTION = 0.10
RESUME_REWARM_STEPS = 20
EVAL_EVERY = int(os.environ.get("HOME_DECOR_EVAL_EVERY", "50"))
TEXT_LENGTH = 64
VAL_ROWS = 1_000
STL_VAL_QUERIES = int(os.environ.get("HOME_DECOR_STL_VAL_QUERIES", "500"))
STL_VAL_GALLERY = None
NUM_WORKERS = min(16, max(2, (os.cpu_count() or 4) - 2))
MAX_HOURS = float(os.environ.get("HOME_DECOR_MAX_HOURS", "100.0"))

IMAGE_PAIR_WEIGHT = float(os.environ.get("HOME_DECOR_PAIR_WEIGHT", "1.0"))
PAIR_LOSS = os.environ.get("HOME_DECOR_PAIR_LOSS", "infonce").strip().lower()
INFONCE_TAU = float(os.environ.get("HOME_DECOR_INFONCE_TAU", "0.07"))
STOP_AT_STEP = int(os.environ.get("HOME_DECOR_STOP_AT_STEP", "0"))
DISTILL_WEIGHT = float(os.environ.get("HOME_DECOR_DISTILL_WEIGHT", "0.05"))

# Augmentation: simulates the catalog -> real-room-photo domain gap that the
# training data does not contain. 0.0 disables, 1.0 is the tuned default.
AUG_STRENGTH = float(os.environ.get("HOME_DECOR_AUG", "1.0"))

# Per-step field subsampling. Title/caption/category always kept; the rest are
# sampled so the text tower does not run 12 x EFF_BATCH sequences per step.
FIELD_KEEP_ALWAYS = ("title", "vlm_caption", "rc_caption", "category")
FIELD_SAMPLE_K = int(os.environ.get("HOME_DECOR_FIELD_SAMPLE_K", "3"))

# Optional in-batch hard negatives: concentrate each batch on a few categories.
HARD_NEG_FRACTION = float(os.environ.get("HOME_DECOR_HARD_NEG", "0.0"))
HARD_NEG_CATEGORIES = int(os.environ.get("HOME_DECOR_HARD_NEG_CATS", "12"))

LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]

if PAIR_LOSS not in {"siglip", "infonce"}:
    raise ValueError(f"HOME_DECOR_PAIR_LOSS must be 'siglip' or 'infonce', got {PAIR_LOSS!r}")
if IMAGE_PAIR_WEIGHT < 0:
    raise ValueError("HOME_DECOR_PAIR_WEIGHT must be non-negative")
if INFONCE_TAU <= 0:
    raise ValueError("HOME_DECOR_INFONCE_TAU must be positive")
if CHUNK < 1 or EFF_BATCH < CHUNK:
    raise ValueError("HOME_DECOR_CHUNK must be >=1 and <= HOME_DECOR_EFF_BATCH")
if not 0.0 <= HARD_NEG_FRACTION <= 1.0:
    raise ValueError("HOME_DECOR_HARD_NEG must be in [0, 1]")

# Run B mix. Extra sources take their share from it proportionally (override with HOME_DECOR_MIX as JSON).
# Open Images (real photos): same 15% share as pix3d + coco had in v3-real_photos, so the runs stay comparable
EXTRA_MIX = {"pix3d": 0.10, "coco_scenes": 0.05, "openimages_crops": 0.10, "openimages_scenes": 0.05}
SOURCE_MIX_OVERRIDE = {"abo": 0.45, "deepfurniture_crops": 0.40, "deepfurniture_scenes": 0.15}
if EXTRA:
    _extra = {s: EXTRA_MIX.get(s, 0.05) for s in EXTRA}
    SOURCE_MIX_OVERRIDE = ({s: p * (1 - sum(_extra.values())) for s, p in SOURCE_MIX_OVERRIDE.items()} | _extra)
if os.environ.get("HOME_DECOR_MIX"):
    SOURCE_MIX_OVERRIDE = json.loads(os.environ["HOME_DECOR_MIX"])
# rc_caption / rc_query: re-captions written from the image (data/recaption.py, DF scenes and crops only;
# empty elsewhere, so other experiments are unchanged). rc_query = one of 3 search queries, rotated per draw.
FIELD_WEIGHTS = {
    "title": 1.0, "vlm_caption": 1.0, "rc_caption": 1.0, "rc_query": 0.5,
    "category": 0.1, "category_coarse": 0.05,
    "style": 0.1, "vlm_style": 0.1,
    "material": 0.05, "vlm_materials": 0.05,
    "color": 0.05, "vlm_colors": 0.05,
    "keywords": 0.05, "vlm_details": 0.05,
}
SHORT_FIELDS = set(FIELD_WEIGHTS) - {"title", "vlm_caption", "rc_caption"}   # short: identical texts count as positives
SET_LIKE_FIELDS = {
    "style", "vlm_style", "material", "vlm_materials",
    "color", "vlm_colors", "keywords", "vlm_details",
}
DROP_FIELDS = {
    "abo": {"style"},
    "deepfurniture_crops": {"title"},
    # template titles ("scandinavian room interior with door, ...") repeat across scenes and would be treated as
    # non-matches for each other; scenes are described by rc_caption / rc_query instead (empty if no recaption).
    # Opt-in (HOME_DECOR_SCENE_TITLE=drop), so runs without re-captions keep their only scene text.
    "deepfurniture_scenes": {"title"} if os.environ.get("HOME_DECOR_SCENE_TITLE", "keep") == "drop" else set(),
    # COCO "category" is the largest object only, which mislabels a whole room photo; captions describe it.
    "coco_scenes": {"category", "category_coarse"},
}
# DeepFurniture's own coarse `style` label (modern / classic / chinese ...): shared by about a third of each
# batch, so it dominated the loss (~36%). The re-captions already name the style (91% of scenes, 64% of crops).
# Opt-in (HOME_DECOR_DF_STYLE=drop).
if os.environ.get("HOME_DECOR_DF_STYLE", "keep") == "drop":
    for _source in ("deepfurniture_crops", "deepfurniture_scenes"):
        DROP_FIELDS[_source] = set(DROP_FIELDS.get(_source, set())) | {"style"}

WORK = Path(os.environ.get("HOME_DECOR_WORK", "/tmp/decore/train"))   # local scratch
RUN_DIR = WORK / "runs" / RUN_NAME
STL_ROOT = Path("/tmp/stl-home-eval")
START_TIME = time.time()
Image.MAX_IMAGE_PIXELS = None
AUDIT_ONLY = os.environ.get("HOME_DECOR_AUDIT_ONLY", "0") == "1"

# Where checkpoints, pointers and logs go. "local": run_training.py sets train_v3.FS = fsutil.fs (text-only writes,
# so binary files are stored as base64 parts + a manifest) under HOME_DECOR_MODELS_DIR/<run>/. "hf": the old
# private HF model repo. Local runs keep weights only (bf16, best + latest saved step): no optimizer state.
STORE = os.environ.get("HOME_DECOR_STORE", "hf").strip().lower()
MODELS_DIR = os.environ.get("HOME_DECOR_MODELS_DIR", "").strip()
FS = None
# Checkpoint selection. "multi": validation split only, several data types, no STL (STL stays a test set).
# "stl": the old score (STL R@1/R@10 + DF + style/category).
SELECTION = os.environ.get("HOME_DECOR_SELECTION", "stl").strip().lower()
# Dry run (CPU): a few rows through data loading, text fields, validation, selection, checkpoint save to the
# store, reload and compare; then the dry-run outputs are deleted. No training step.
DRY_RUN = os.environ.get("HOME_DECOR_DRY_RUN", "0") == "1"
if STORE not in {"local", "hf"}:
    raise ValueError("HOME_DECOR_STORE must be 'local' or 'hf'")
if SELECTION not in {"multi", "stl"}:
    raise ValueError("HOME_DECOR_SELECTION must be 'multi' or 'stl'")
if DRY_RUN:
    RUN_NAME = f"{RUN_NAME}-dryrun"
    RUN_DIR = WORK / "runs" / RUN_NAME

# ------------------------------------------------------------------ auth


def get_token():
    token = os.environ.get("HF_TOKEN")
    if not token:
        from huggingface_hub import get_token as get_cached_hf_token
        token = get_cached_hf_token()
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
        except ImportError:
            pass
    if not token:
        raise RuntimeError("HF_TOKEN is not set and no cached Hugging Face login was found")
    return token


from huggingface_hub import HfApi, hf_hub_download, login, snapshot_download
TOKEN = None

# ------------------------------------------------------------------ text helpers


def text_value(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, dict):
        return text_value(value.get("value", value.get("name", "")))
    if isinstance(value, str):
        value = value.strip()
        if value.startswith(("[", "{")):
            try:
                return text_value(json.loads(value))
            except (json.JSONDecodeError, TypeError):
                pass
        return " ".join(value.replace("_", " ").split())
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            item = text_value(item)
            if item and item.casefold() not in {x.casefold() for x in out}:
                out.append(item)
        return ", ".join(out)
    return " ".join(str(value).replace("_", " ").split())


def field_text(field, value):
    value = text_value(value).lower()
    if not value or value in {"none", "unknown", "n/a", "[]"}:
        return ""
    if field in {"style", "vlm_style"}:
        return f"{value} style"
    if field in {"material", "vlm_materials"}:
        return f"{value} material"
    if field in {"category", "category_coarse"}:
        return f"{'an' if value[:1] in 'aeiou' else 'a'} {value}"
    return value


def label_key(field, value):
    value = value.casefold().strip()
    if field not in SET_LIKE_FIELDS:
        return value
    suffix = " style" if field in {"style", "vlm_style"} else (
        " material" if field in {"material", "vlm_materials"} else ""
    )
    if suffix and value.endswith(suffix):
        value = value[:-len(suffix)]
    return tuple(sorted({part.strip() for part in value.split(",") if part.strip()}))


def rotating_partner(alternatives, index, sampling_step):
    return alternatives[(SEED + index + sampling_step) % len(alternatives)]


def product_id(row):
    value = row.get("furniture_id") or row.get("identity_id") or row.get("product_id")
    return "" if value is None else str(value)


def identity(row, source, index):
    if source == "deepfurniture_crops" and product_id(row):
        return f"deepfurniture-product:{product_id(row)}"
    for key in ("group_id", "furniture_id", "identity_id", "product_id", "scene_id", "image_id", "id"):
        value = row.get(key)
        if value not in (None, ""):
            prefix = "deepfurniture-product" if key in {"furniture_id", "identity_id"} else source
            return f"{prefix}:{value}"
    return f"{source}:row:{index}"


def sampled_fields(step):
    optional = [f for f in FIELD_WEIGHTS if f not in FIELD_KEEP_ALWAYS]
    if FIELD_SAMPLE_K >= len(optional):
        return set(FIELD_WEIGHTS)
    rng = random.Random(SEED * 7919 + step)
    return set(FIELD_KEEP_ALWAYS) | set(rng.sample(optional, FIELD_SAMPLE_K))

# ------------------------------------------------------------------ image helpers


def image_value(row):
    for key in ("image", "crop", "scene", "render"):
        if row.get(key) is not None:
            return row[key]
    raise KeyError(f"No image-like column; columns={sorted(row)}")


def flatten(image):
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return image.convert("RGB")


def as_pil(value):
    if isinstance(value, Image.Image):
        result = value
    elif isinstance(value, dict) and value.get("bytes") is not None:
        result = Image.open(io.BytesIO(value["bytes"]))
    elif isinstance(value, dict) and value.get("path"):
        result = Image.open(value["path"])
    elif isinstance(value, (str, Path)):
        result = Image.open(value)
    else:
        raise TypeError(f"Unsupported image value {type(value)!r}")
    return flatten(ImageOps.exif_transpose(result))


def open_scaled(path, max_side):
    image = Image.open(path)
    image.draft("RGB", (max_side, max_side))
    image = flatten(ImageOps.exif_transpose(image))
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side))
    return image


def domain_augment(image, rng, strength, heavy):
    # Applied to the anchor only. "heavy" is for clean catalog renders, which
    # need the full room-photo simulation; scene crops get a lighter pass.
    if strength <= 0:
        return image
    s = strength

    if rng.random() < 0.9 * s:
        w, h = image.size
        scale = rng.uniform(0.45 if heavy else 0.65, 1.0)
        ratio = math.exp(rng.uniform(math.log(0.78), math.log(1.28)))
        cw = min(w, int(round(math.sqrt(w * h * scale * ratio))))
        ch = min(h, int(round(math.sqrt(w * h * scale / ratio))))
        x0 = rng.randint(0, max(0, w - cw))
        y0 = rng.randint(0, max(0, h - ch))
        image = image.crop((x0, y0, x0 + cw, y0 + ch))

    if heavy and rng.random() < 0.5 * s:
        w, h = image.size
        d = 0.12 * s
        src = [(0, 0), (w, 0), (w, h), (0, h)]
        dst = [(x + rng.uniform(-d, d) * w, y + rng.uniform(-d, d) * h) for x, y in src]
        try:
            coeffs = perspective_coeffs(dst, src)
            image = image.transform((w, h), Image.PERSPECTIVE, coeffs,
                                    Image.BICUBIC, fillcolor=(255, 255, 255))
        except Exception:
            pass

    if rng.random() < 0.8 * s:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.65, 1.35))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.70, 1.30))
        image = ImageEnhance.Color(image).enhance(rng.uniform(0.60, 1.40))

    if heavy and rng.random() < 0.35 * s:
        # Warm/cool indoor lighting cast.
        tint = Image.new("RGB", image.size, (
            rng.randint(200, 255), rng.randint(190, 255), rng.randint(170, 255)))
        image = Image.blend(image, tint, rng.uniform(0.04, 0.16))

    if rng.random() < 0.3 * s:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.4)))

    if rng.random() < 0.5:
        image = ImageOps.mirror(image)

    if heavy and rng.random() < 0.3 * s:
        w, h = image.size
        ow, oh = int(w * rng.uniform(0.12, 0.33)), int(h * rng.uniform(0.12, 0.33))
        x0, y0 = rng.randint(0, max(0, w - ow)), rng.randint(0, max(0, h - oh))
        patch = Image.new("RGB", (ow, oh), (
            rng.randint(60, 200), rng.randint(60, 200), rng.randint(60, 200)))
        image.paste(patch, (x0, y0))

    if rng.random() < 0.4 * s:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=rng.randint(32, 78))
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")

    return image


def perspective_coeffs(source, target):
    matrix = []
    for (sx, sy), (tx, ty) in zip(source, target):
        matrix.append([sx, sy, 1, 0, 0, 0, -tx * sx, -tx * sy])
        matrix.append([0, 0, 0, sx, sy, 1, -ty * sx, -ty * sy])
    a = np.array(matrix, dtype=np.float64)
    b = np.array(target, dtype=np.float64).reshape(8)
    return np.linalg.solve(a, b).tolist()

# ------------------------------------------------------------------ data pipeline


class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, render_index, abo_views, pix3d_views=None):
        self.datasets, self.render_index, self.abo_views = datasets, render_index, abo_views
        self.pix3d_views = pix3d_views or {}
        self.offsets, offset = {}, 0
        for source in SOURCES:
            self.offsets[source], offset = offset, offset + len(datasets[source])
        self.length = offset

    def __len__(self):
        return self.length

    def global_index(self, source, local):
        return self.offsets[source] + local

    def locate(self, index):
        for source in reversed(SOURCES):
            if index >= self.offsets[source]:
                return source, index - self.offsets[source]
        raise IndexError(index)

    def __getitem__(self, index):
        sampling_step = 0
        if isinstance(index, tuple):
            index, sampling_step = index
        source, local = self.locate(index)
        row = self.datasets[source][local]
        pid = product_id(row)
        dropped = DROP_FIELDS.get(source, set())
        keep = sampled_fields(sampling_step)
        partner, kind = None, ""
        if IMAGE_PAIR_WEIGHT > 0 and source == "deepfurniture_crops" and pid in self.render_index:
            partner = as_pil(image_value(self.datasets["deepfurniture_products"][self.render_index[pid]]))
            kind = "crop_render"
        elif IMAGE_PAIR_WEIGHT > 0 and source == "abo" and self.abo_views.get(local):
            alternatives = self.abo_views[local]
            other = rotating_partner(alternatives, index, sampling_step)
            partner = as_pil(image_value(self.datasets["abo"][other]))
            kind = "abo_view"
        elif IMAGE_PAIR_WEIGHT > 0 and source == "pix3d" and self.pix3d_views.get(local):
            # another real photo of the same 3D model
            other = rotating_partner(self.pix3d_views[local], index, sampling_step)
            partner = as_pil(image_value(self.datasets["pix3d"][other]))
            kind = "pix3d_view"

        image = as_pil(image_value(row))
        rng = random.Random(SEED + index * 131071 + sampling_step)
        # Catalog renders need the heavy pass; DF scene crops are already cluttered.
        image = domain_augment(image, rng, AUG_STRENGTH, heavy=(source == "abo"))

        fields = {f: ("" if (f in dropped or f not in keep) else field_text(f, row.get(f))) for f in FIELD_WEIGHTS}
        if source == "coco_scenes" and row.get("captions"):
            # 5 human captions per photo: use a different one each time the photo is drawn
            captions = row["captions"]
            fields["title"] = field_text("title", captions[(index + sampling_step) % len(captions)])
        queries = row.get("rc_queries") or []
        if queries and "rc_query" in keep and "rc_query" not in dropped:
            # 3 search queries per image: use a different one each time the image is drawn
            fields["rc_query"] = field_text("rc_query", queries[(index + sampling_step) % len(queries)])
        return {
            "source": source, "row_index": local, "product_id": pid,
            "group": identity(row, source, local), "image": image,
            "fields": fields, "partner": partner, "partner_kind": kind,
        }


class UniqueBatchSampler:
    def __init__(self, dataset, groups, weights, probabilities, start, stop, categories=None):
        self.dataset, self.groups, self.weights = dataset, groups, weights
        self.probabilities, self.start, self.stop = probabilities, start, stop
        self.cum = {s: list(itertools.accumulate(weights[s])) for s in SOURCES}
        self.categories = categories or {}
        self.by_category = {}
        for source, labels in self.categories.items():
            buckets = defaultdict(list)
            for local, label in enumerate(labels):
                if label:
                    buckets[label].append(local)
            self.by_category[source] = {k: v for k, v in buckets.items() if len(v) >= 8}

    def __len__(self):
        return self.stop - self.start

    def counts(self):
        raw = {s: EFF_BATCH * self.probabilities[s] for s in SOURCES}
        counts = {s: int(raw[s]) for s in SOURCES}
        for s in sorted(SOURCES, key=lambda s: raw[s] - counts[s], reverse=True)[:EFF_BATCH - sum(counts.values())]:
            counts[s] += 1
        return counts

    def draw(self, source, quota, rng, used, chosen):
        population, attempts = range(len(self.groups[source])), 0
        pool = None
        if HARD_NEG_FRACTION > 0 and self.by_category.get(source) and rng.random() < HARD_NEG_FRACTION:
            names = rng.sample(sorted(self.by_category[source]),
                               min(HARD_NEG_CATEGORIES, len(self.by_category[source])))
            pool = [local for name in names for local in self.by_category[source][name]]
        while quota:
            if pool:
                batch = rng.choices(pool, k=max(8, 2 * quota))
            else:
                batch = rng.choices(population, cum_weights=self.cum[source], k=max(8, 2 * quota))
            for local in batch:
                if not quota:
                    break
                attempts += 1
                if self.groups[source][local] in used:
                    if attempts > EFF_BATCH * 100:
                        if pool:
                            pool = None
                            attempts = 0
                            break
                        raise RuntimeError(f"Not enough unique {source} groups for a batch")
                    continue
                used.add(self.groups[source][local])
                chosen.append(self.dataset.global_index(source, local))
                quota -= 1

    def __iter__(self):
        counts = self.counts()
        for step in range(self.start, self.stop):
            rng, chosen, used = random.Random(SEED + step), [], set()
            for source in SOURCES:
                self.draw(source, counts[source], rng, used, chosen)
            rng.shuffle(chosen)
            yield [(index, step) for index in chosen]


class Collator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, items):
        pixels = self.processor(images=[x["image"] for x in items], return_tensors="pt").pixel_values
        texts, entries = [], {}
        for field in FIELD_WEIGHTS:
            entries[field] = []
            for image_index, item in enumerate(items):
                value = item["fields"][field]
                if value:
                    entries[field].append((image_index, len(texts), value))
                    texts.append(value)
        tokens = self.processor(text=texts, padding="max_length", truncation=True,
                                max_length=TEXT_LENGTH, return_tensors="pt")
        partners = [(i, x["partner"], x["partner_kind"]) for i, x in enumerate(items) if x["partner"] is not None]
        return {
            "pixel_values": pixels, "input_ids": tokens.input_ids,
            "attention_mask": tokens.get("attention_mask"), "entries": entries,
            "partner_pixels": (self.processor(images=[p[1] for p in partners], return_tensors="pt").pixel_values
                               if partners else None),
            "partner_indices": torch.tensor([p[0] for p in partners], dtype=torch.long),
            "partner_kinds": [p[2] for p in partners],
            "metadata": [{k: x[k] for k in ("source", "row_index", "product_id", "group", "partner_kind")} |
                         {"fields": {k: v for k, v in x["fields"].items() if v}} for x in items],
            "preview_images": [x["image"] for x in items[:8]],
        }

# ------------------------------------------------------------------ loss


def pooled(value):
    return value if hasattr(value, "shape") else value.pooler_output


def siglip_loss(query, gallery, positive, scale, bias):
    logits = (query.float() @ gallery.float().T) * scale.float() + bias.float()
    labels = positive.to(logits.dtype).mul(2).sub(1)
    return -F.logsigmoid(labels * logits).sum() / query.shape[0]


def info_nce(a, b, tau=INFONCE_TAU):
    logits = (a.float() @ b.float().T) / tau
    labels = torch.arange(len(a), device=a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def representation_loss(model, batch, images, text, partner, teacher_feats, device):
    scale = model.logit_scale.float().exp().clamp(max=100)
    bias = model.logit_bias.float() if hasattr(model, "logit_bias") else torch.zeros((), device=device)

    total, active, parts = torch.zeros((), device=device), 0.0, {}
    for field, entries in batch["entries"].items():
        if len(entries) < 2:
            continue
        ii = torch.tensor([x[0] for x in entries], device=device)
        ti = torch.tensor([x[1] for x in entries], device=device)
        keys = [label_key(field, x[2]) for x in entries]
        positive = (torch.tensor([[a == b for b in keys] for a in keys], device=device)
                    if field in SHORT_FIELDS else torch.eye(len(entries), dtype=torch.bool, device=device))
        parts[field] = siglip_loss(text[ti], images[ii], positive, scale, bias)
        total = total + FIELD_WEIGHTS[field] * parts[field]
        active += FIELD_WEIGHTS[field]
    if not active:
        raise RuntimeError("No trainable text fields in batch")
    total = total / active
    text_loss = total.detach()
    contributions = {
        field: float((FIELD_WEIGHTS[field] * value.detach() / active))
        for field, value in parts.items()
    }

    pair_loss = torch.zeros((), device=device)
    pair_parts = {}
    if partner is not None and len(partner) >= 2:
        all_indices = batch["partner_indices"].to(device)
        for kind in ("crop_render", "abo_view", "pix3d_view"):
            positions = [n for n, value in enumerate(batch["partner_kinds"]) if value == kind]
            if len(positions) < 2:
                continue
            positions = torch.tensor(positions, device=device)
            anchor = images[all_indices[positions]]
            target = partner[positions]
            positive = torch.eye(len(anchor), dtype=torch.bool, device=device)
            if PAIR_LOSS == "infonce":
                pair_parts[kind] = info_nce(anchor, target)
            else:
                pair_parts[kind] = (
                    siglip_loss(anchor, target, positive, scale, bias)
                    + siglip_loss(target, anchor, positive, scale, bias)
                ) / 2
        if pair_parts:
            pair_loss = torch.stack(list(pair_parts.values())).mean()
            total = total + IMAGE_PAIR_WEIGHT * pair_loss

    distill = torch.zeros((), device=device)
    if DISTILL_WEIGHT and teacher_feats is not None:
        distill = (1 - (images * teacher_feats).sum(-1)).mean()
        total = total + DISTILL_WEIGHT * distill

    kinds = Counter(batch["partner_kinds"])
    metrics = {
        "field_raw": {k: float(v.detach()) for k, v in parts.items()},
        "field_contribution": contributions,
        "pair_by_kind": {k: float(v.detach()) for k, v in pair_parts.items()},
        "pair": float(pair_loss.detach()), "distill": float(distill.detach()),
        "text_loss": float(text_loss),
        "weighted_pair": float(IMAGE_PAIR_WEIGHT * pair_loss.detach()),
        "weighted_distill": float(DISTILL_WEIGHT * distill.detach()),
        "logit_bias": float(bias.detach()), "effective_scale": float(scale.detach()),
        "pairs_crop_render": kinds.get("crop_render", 0),
        "pairs_abo_view": kinds.get("abo_view", 0),
        "pairs_pix3d_view": kinds.get("pix3d_view", 0),
        "n_texts": int(batch["input_ids"].shape[0]),
    }
    return total, metrics

# ------------------------------------------------------------------ gradcache


def rng_snapshot():
    return torch.get_rng_state(), torch.cuda.get_rng_state()


def rng_restore(state):
    torch.set_rng_state(state[0])
    torch.cuda.set_rng_state(state[1])


def encode_chunks(fn, total, chunk, dtype, collect_rng):
    out, states = [], []
    with torch.no_grad():
        for start in range(0, total, chunk):
            if collect_rng:
                states.append(rng_snapshot())
            with torch.autocast("cuda", dtype=dtype):
                feats = pooled(fn(start, min(start + chunk, total)))
            out.append(F.normalize(feats.float(), dim=-1))
    return torch.cat(out), states


def backprop_chunks(fn, total, chunk, dtype, grads, states):
    for n, start in enumerate(range(0, total, chunk)):
        stop = min(start + chunk, total)
        rng_restore(states[n])
        with torch.autocast("cuda", dtype=dtype):
            feats = pooled(fn(start, stop))
        feats = F.normalize(feats.float(), dim=-1)
        feats.backward(gradient=grads[start:stop])


def gradcache_step(model, teacher, batch, device, dtype, chunk):
    pixels = batch["pixel_values"].to(device, non_blocking=True)
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    mask = batch["attention_mask"]
    mask = mask.to(device, non_blocking=True) if mask is not None else None
    partner_pixels = batch["partner_pixels"]
    if partner_pixels is not None:
        partner_pixels = partner_pixels.to(device, non_blocking=True)

    def image_fn(a, b):
        return model.get_image_features(pixel_values=pixels[a:b])

    def partner_fn(a, b):
        return model.get_image_features(pixel_values=partner_pixels[a:b])

    def text_fn(a, b):
        kwargs = {"input_ids": input_ids[a:b]}
        if mask is not None:
            kwargs["attention_mask"] = mask[a:b]
        return model.get_text_features(**kwargs)

    # Phase 1: embeddings without a graph.
    img_e, img_states = encode_chunks(image_fn, len(pixels), chunk, dtype, True)
    txt_e, txt_states = encode_chunks(text_fn, len(input_ids), chunk * 4, dtype, True)
    par_e, par_states = (None, [])
    if partner_pixels is not None and len(partner_pixels) >= 2:
        par_e, par_states = encode_chunks(partner_fn, len(partner_pixels), chunk, dtype, True)

    teacher_feats = None
    if DISTILL_WEIGHT:
        if USE_LORA:
            with model.disable_adapter():
                teacher_feats, _ = encode_chunks(image_fn, len(pixels), chunk, dtype, False)
        else:
            def teacher_fn(a, b):
                return teacher.get_image_features(pixel_values=pixels[a:b])
            teacher_feats, _ = encode_chunks(teacher_fn, len(pixels), chunk, dtype, False)

    img_e = img_e.detach().requires_grad_(True)
    txt_e = txt_e.detach().requires_grad_(True)
    if par_e is not None:
        par_e = par_e.detach().requires_grad_(True)

    # Phase 2: full-batch loss over cached embeddings. This also deposits
    # gradients on logit_scale / logit_bias, which live outside the towers.
    loss, metrics = representation_loss(model, batch, img_e, txt_e, par_e, teacher_feats, device)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite representation loss: {loss}")
    loss.backward()

    # Phase 3: re-run each chunk with a graph and inject the cached gradients.
    backprop_chunks(image_fn, len(pixels), chunk, dtype, img_e.grad, img_states)
    backprop_chunks(text_fn, len(input_ids), chunk * 4, dtype, txt_e.grad, txt_states)
    if par_e is not None:
        backprop_chunks(partner_fn, len(partner_pixels), chunk, dtype, par_e.grad, par_states)

    return float(loss.detach()), metrics


def build_optimizer(model):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "logit" in name or "norm" in name.lower() or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY},
                              {"params": no_decay, "weight_decay": 0.0}], lr=LR)


def source_state(datasets):
    groups, weights, masses = {}, {}, {}
    for source in SOURCES:
        id_columns = [n for n in ("group_id", "furniture_id", "identity_id", "product_id",
                                  "scene_id", "image_id", "id") if n in datasets[source].column_names]
        metadata = datasets[source].select_columns(id_columns)
        groups[source] = [identity(metadata[i], source, i) for i in range(len(metadata))]
        if source == "deepfurniture_crops":
            counts = Counter(groups[source])
            weights[source] = [1 / counts[g] for g in groups[source]]
        else:
            weights[source] = [1.0] * len(groups[source])
        masses[source] = sum(weights[source])
    chosen = SOURCE_MIX_OVERRIDE or masses
    denominator = sum(chosen.values())
    return groups, weights, masses, {s: chosen[s] / denominator for s in SOURCES}


def category_labels(datasets):
    labels = {}
    for source in SOURCES:
        column = next((c for c in ("category_coarse", "category") if c in datasets[source].column_names), None)
        if column is None:
            labels[source] = []
            continue
        labels[source] = [text_value(v).lower() for v in datasets[source][column]]
    return labels


def select(dataset, limit, seed):
    if not limit or len(dataset) <= limit:
        return dataset
    ids = list(range(len(dataset)))
    random.Random(seed).shuffle(ids)
    return dataset.select(sorted(ids[:limit]))


def log(path, **row):
    row = {"time": round(time.time(), 3), "hours": round((time.time() - START_TIME) / 3600, 2), **row}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")
    print(json.dumps(row, default=str), flush=True)


def batch_preview(batch, path):
    width, ih, th = 384, 256, 170
    rows = min(8, len(batch["preview_images"]))
    sheet = Image.new("RGB", (width, rows * (ih + th)), "white")
    draw = ImageDraw.Draw(sheet)
    for n, (img, meta) in enumerate(zip(batch["preview_images"], batch["metadata"][:rows])):
        img = img.copy(); img.thumbnail((width, ih)); top = n * (ih + th)
        sheet.paste(img, ((width - img.width) // 2, top))
        label = f"{meta['source']} | {meta['group']} | partner: {meta['partner_kind'] or '-'}\n" + "\n".join(
            f"{k}: {v}" for k, v in list(meta["fields"].items())[:6])
        draw.multiline_text((6, top + ih + 4), label[:900], fill="black", spacing=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)

# ------------------------------------------------------------------ evaluation


def embed_images(model, processor, values, device, dtype, batch_size=64):
    def load(x):
        return as_pil(x() if callable(x) else x)
    output = []; model.eval()
    with torch.no_grad(), ThreadPoolExecutor(4) as pool:
        for start in range(0, len(values), batch_size):
            batch = list(pool.map(load, values[start:start + batch_size]))
            pixels = processor(images=batch, return_tensors="pt").pixel_values.to(device)
            with torch.autocast("cuda", dtype=dtype):
                output.append(F.normalize(pooled(model.get_image_features(pixel_values=pixels)).float(), dim=-1).cpu())
    return torch.cat(output)


def embed_texts(model, processor, values, device, dtype, batch_size=128):
    output = []; model.eval()
    values = [v.lower() for v in values]
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            inputs = processor(text=values[start:start + batch_size], padding="max_length",
                               truncation=True, max_length=TEXT_LENGTH, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.autocast("cuda", dtype=dtype):
                output.append(F.normalize(pooled(model.get_text_features(**inputs)).float(), dim=-1).cpu())
    return torch.cat(output)


def retrieval(query, gallery, relevant, ks=(1, 10)):
    similarities, ranks = query @ gallery.T, []
    for n, choices in enumerate(relevant):
        if choices:
            best = similarities[n, torch.tensor(sorted(choices))].max()
            ranks.append(int((similarities[n] > best).sum()) + 1)
    ranks = np.asarray(ranks)
    if not len(ranks):
        raise RuntimeError("Retrieval evaluation has no queries with relevant gallery items")
    result = {f"R@{k}": float((ranks <= k).mean()) for k in ks}
    return result | {"MRR": float((1 / ranks).mean()), "median_rank": float(np.median(ranks)), "queries": len(ranks)}


def multi_relevant_retrieval(query, gallery, relevant, k=10):
    similarities = query @ gallery.T
    ndcg, recall, reciprocal = [], [], []
    for n, choices in enumerate(relevant):
        if not choices:
            continue
        order = torch.argsort(similarities[n], descending=True)
        truth = torch.zeros(len(gallery), dtype=torch.float32)
        truth[list(choices)] = 1
        ranked = truth[order]
        top = ranked[:k]
        discounts = 1 / torch.log2(torch.arange(2, len(top) + 2, dtype=torch.float32))
        dcg = float((top * discounts).sum())
        ideal = float(discounts[:min(k, len(choices))].sum())
        ndcg.append(dcg / ideal if ideal else 0.0)
        recall.append(float(top.sum()) / len(choices))
        first = torch.nonzero(ranked, as_tuple=False)
        reciprocal.append(1 / (int(first[0]) + 1) if len(first) else 0.0)
    if not ndcg:
        return {"nDCG@10": 0.0, "R@10": 0.0, "MRR": 0.0, "queries": 0}
    return {
        "nDCG@10": float(np.mean(ndcg)), "R@10": float(np.mean(recall)),
        "MRR": float(np.mean(reciprocal)), "queries": len(ndcg), "gallery": len(gallery),
    }


def attribute_eval(model, processor, dataset, device, dtype, limit, seed):
    dataset = select(dataset, limit, seed)
    selected = {}
    for n in range(len(dataset)):
        row = dataset[n]
        group = identity(row, "abo", n)
        if group not in selected or bool(row.get("is_main", False)):
            selected[group] = row
    rows = list(selected.values())
    gallery = embed_images(model, processor, [image_value(row) for row in rows], device, dtype)
    field_groups = {
        "category": ("category", "category_coarse"),
        "style": ("style", "vlm_style"),
        "material": ("material", "vlm_materials"),
        "color": ("color", "vlm_colors"),
    }
    result = {}
    for name, fields in field_groups.items():
        mapping = defaultdict(set)
        for index, row in enumerate(rows):
            for field in fields:
                value = field_text(field, row.get(field))
                if value:
                    mapping[value].add(index)
        queries = sorted(value for value, matches in mapping.items() if len(matches) >= 2)
        if len(queries) > 200:
            random.Random(seed + len(name)).shuffle(queries)
            queries = sorted(queries[:200])
        if not queries:
            result[name] = {"nDCG@10": 0.0, "R@10": 0.0, "MRR": 0.0, "queries": 0,
                            "gallery": len(rows)}
            continue
        result[name] = multi_relevant_retrieval(
            embed_texts(model, processor, queries, device, dtype), gallery,
            [mapping[value] for value in queries], k=10,
        )
    return result


def text_image_eval(model, processor, dataset, device, dtype, limit, seed):
    dataset = select(dataset, limit, seed)
    images, image_groups, group_text = [], [], {}
    for n in range(len(dataset)):
        row = dataset[n]; group = identity(row, "validation", n)
        text = field_text("title", row.get("title")) or field_text("category", row.get("category"))
        if text:
            images.append(image_value(row)); image_groups.append(group); group_text.setdefault(group, text)
    groups = list(group_text)
    relevant = [{n for n, v in enumerate(image_groups) if v == g} for g in groups]
    return retrieval(embed_texts(model, processor, [group_text[x] for x in groups], device, dtype),
                     embed_images(model, processor, images, device, dtype), relevant)


def crop_render_eval(model, processor, crops, products, device, dtype, limit, seed):
    crops = select(crops, limit, seed)
    pids = [str(x) for x in products["furniture_id"]]
    index = {}
    for n, pid in enumerate(pids):
        index.setdefault(pid, n)
    keep_rows = sorted(index.values())
    remap = {old: new for new, old in enumerate(keep_rows)}
    queries, relevant = [], []
    crop_ids = [str(x) for x in crops["furniture_id"]]
    has = crops["has_product"] if "has_product" in crops.column_names else [True] * len(crops)
    for n, (pid, h) in enumerate(zip(crop_ids, has)):
        if pid in index and h:
            queries.append(image_value(crops[n])); relevant.append({remap[index[pid]]})
    gallery = [image_value(products[i]) for i in keep_rows]
    return retrieval(embed_images(model, processor, queries, device, dtype),
                     embed_images(model, processor, gallery, device, dtype), relevant, ks=(1, 10))


def ensure_stl():
    if not (STL_ROOT / "queries.jsonl").exists():
        snapshot_download(STL_REPO, repo_type="dataset", token=TOKEN, local_dir=STL_ROOT,
                          allow_patterns=["queries.jsonl", "gallery.jsonl", "images.tar"])
    if not (STL_ROOT / "scenes").exists() or not (STL_ROOT / "products").exists():
        with tarfile.open(STL_ROOT / "images.tar") as tar:
            tar.extractall(STL_ROOT, filter="data")
        (STL_ROOT / "images.tar").unlink(missing_ok=True)
    return STL_ROOT


def stl_eval(model, processor, device, dtype, count, smoke, dump_path=None):
    root = ensure_stl()
    queries = [json.loads(x) for x in (root / "queries.jsonl").read_text().splitlines() if x]
    gallery = [json.loads(x) for x in (root / "gallery.jsonl").read_text().splitlines() if x]
    rng = random.Random(SEED); queries = rng.sample(queries, min(count, len(queries)))
    cap = 1_024 if smoke else STL_VAL_GALLERY
    if cap and len(gallery) > cap:
        required = {p for q in queries for p in q["relevant"]}
        keep = [x for x in gallery if x["product"] in required]
        other = [x for x in gallery if x["product"] not in required]; rng.shuffle(other)
        gallery = keep + other[:max(0, cap - len(keep))]
    index = {x["product"]: n for n, x in enumerate(gallery)}

    def crop(q):
        def load():
            l, t, r, b = q["bbox"]
            need = int(2 * 224 / max(min(r - l, b - t), 1e-3))
            image = open_scaled(root / "scenes" / f"{q['scene']}.jpg", min(max(need, 448), 4096))
            w, h = image.size
            px, py = (r - l) * .1, (b - t) * .1
            return image.crop((max(0, (l - px) * w), max(0, (t - py) * h),
                               min(w, (r + px) * w), min(h, (b + py) * h)))
        return load

    def product(sig):
        return lambda: open_scaled(root / "products" / f"{sig}.jpg", 448)

    relevant = [{index[p] for p in q["relevant"] if p in index} for q in queries]
    qe = embed_images(model, processor, [crop(q) for q in queries], device, dtype)
    ge = embed_images(model, processor, [product(x["product"]) for x in gallery], device, dtype)
    result = retrieval(qe, ge, relevant, ks=(1, 5, 10, 50, 100))

    # Top-10 dump for the duplicate-product audit. Costs nothing extra.
    if dump_path is not None:
        top = torch.topk(qe @ ge.T, k=min(10, len(gallery)), dim=1).indices
        records = []
        for n, q in enumerate(queries):
            records.append({
                "scene": q["scene"], "bbox": q["bbox"],
                "truth": sorted(q["relevant"]),
                "predicted": [gallery[int(i)]["product"] for i in top[n]],
                "hit_at_1": bool(int(top[n][0]) in relevant[n]),
            })
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text("\n".join(json.dumps(r) for r in records))
    return result | {"gallery": len(gallery)}


def query_eval(model, processor, dataset, device, dtype, limit, seed, main_only=False):
    """New-style search queries written for each validation image (rc_queries, data/recaption.py with
    SPLIT='validation') -> that image. One correct image per query, so generic queries are hard for every model
    alike. Returns None when the split has no queries (old data without rc_queries)."""
    if "rc_queries" not in dataset.column_names:
        return None
    columns = [c for c in ("rc_queries", "is_main", "title", "area_ratio", "in_image_ratio") if c in dataset.column_names]
    meta = dataset.select_columns(columns).to_dict()
    n_rows = len(dataset)
    # Label-quality rules, the same as for train and decided before any results (not model-based): skip swatch
    # listings, and crops too small (< 1% of the room) or too cut off (< 80% inside the image) to show the item.
    area, inside = meta.get("area_ratio") or [1.0] * n_rows, meta.get("in_image_ratio") or [1.0] * n_rows
    keep = [n for n, queries in enumerate(meta["rc_queries"])
            if queries and (not main_only or bool(meta.get("is_main", [True] * n_rows)[n]))
            and "swatch" not in str((meta.get("title") or [""] * n_rows)[n]).lower()
            and (area[n] if area[n] is not None else 1.0) >= 0.01 and (inside[n] if inside[n] is not None else 1.0) >= 0.8]
    if limit and len(keep) > limit:
        keep = sorted(random.Random(seed).sample(keep, limit))
    if not keep:
        return None
    texts, relevant = [], []
    for position, n in enumerate(keep):
        for query in meta["rc_queries"][n]:
            if query:
                texts.append(query); relevant.append({position})
    images = [image_value(dataset[n]) for n in keep]
    return retrieval(embed_texts(model, processor, texts, device, dtype),
                     embed_images(model, processor, images, device, dtype), relevant, ks=(1, 10)) | {"gallery": len(keep)}


def validate(model, processor, datasets, device, dtype, smoke, dump_path=None):
    limit = 64 if DRY_RUN else (256 if smoke else VAL_ROWS)
    if SELECTION == "multi":
        # validation split only; no STL (STL is a test set). Scenes and ABO main photos are small enough to use
        # all of them; crops are sampled.
        small = limit if (smoke or DRY_RUN) else None
        results = {
            "abo_title_to_image": text_image_eval(model, processor, datasets["abo"], device, dtype, limit, SEED),
            "df_crop_to_render": crop_render_eval(model, processor, datasets["deepfurniture_crops"],
                                                  datasets["deepfurniture_products"], device, dtype, limit, SEED),
            "attributes": attribute_eval(model, processor, datasets["abo"], device, dtype, limit, SEED + 2),
            "query_scenes": query_eval(model, processor, datasets["deepfurniture_scenes"], device, dtype,
                                       small, SEED + 3),
            "query_crops": query_eval(model, processor, datasets["deepfurniture_crops"], device, dtype,
                                      limit, SEED + 4),
            "query_abo": query_eval(model, processor, datasets["abo"], device, dtype, small, SEED + 5,
                                    main_only=True),
        }
        model.train()
        gc.collect(); torch.cuda.empty_cache()
        return results
    results = {
        "abo_title_to_image": text_image_eval(model, processor, datasets["abo"], device, dtype, limit, SEED),
        "df_crop_to_render": crop_render_eval(model, processor, datasets["deepfurniture_crops"],
                                              datasets["deepfurniture_products"], device, dtype, limit, SEED),
        "df_scene_text_to_image": text_image_eval(model, processor, datasets["deepfurniture_scenes"],
                                                  device, dtype, limit, SEED + 1),
        "attributes": attribute_eval(model, processor, datasets["abo"], device, dtype, limit, SEED + 2),
        "stl_crop_to_product": stl_eval(model, processor, device, dtype,
                                        64 if smoke else STL_VAL_QUERIES, smoke, dump_path),
    }
    model.train()
    gc.collect(); torch.cuda.empty_cache()
    return results

# ------------------------------------------------------------------ checkpoints


class Store:
    """All run outputs (pointers, logs, previews, model weights) for one run, in the data folder or the HF model repo.

    The store goes through fsutil.fs, which writes text only, so every file is written as base64 parts of 30 MB
    (about 40 MB of text) next to a manifest with its size and SHA-256; reading checks the hash.
    A folder is a set of files plus files.json listing them.
    """
    PART = 30 * 1024 * 1024

    def __init__(self):
        self.local = STORE == "local"
        if self.local:
            if FS is None or not MODELS_DIR:
                raise RuntimeError("HOME_DECOR_STORE=local needs train_v3.FS = fsutil.fs and HOME_DECOR_MODELS_DIR")
            self.root = f"{MODELS_DIR.rstrip('/')}/{RUN_NAME}"
        else:
            self.api = HfApi(token=TOKEN)
            self.api.create_repo(OUTPUT_REPO, repo_type="model", private=True, exist_ok=True)
            self.root = f"runs/{RUN_NAME}"

    def _hf(self, rel):
        return f"{self.root}/{rel}"

    def put_text(self, rel, text):
        if self.local:
            assert FS.put(f"{self.root}/{rel}", text, overwrite=True), rel
        else:
            self.api.upload_file(path_or_fileobj=io.BytesIO(text.encode()), path_in_repo=self._hf(rel),
                                 repo_id=OUTPUT_REPO, repo_type="model")

    def put_json(self, rel, value):
        self.put_text(rel, json.dumps(value, indent=2, default=str))

    def get_text(self, rel):
        """File content, or None if it does not exist."""
        try:
            if self.local:
                path = f"{self.root}/{rel}"
                return FS.head(path, FS.ls(path)[0].size)
            return Path(hf_hub_download(OUTPUT_REPO, self._hf(rel), repo_type="model", token=TOKEN,
                                        force_download=True)).read_text()
        except Exception:
            return None

    def get_json(self, rel):
        text = self.get_text(rel)
        return json.loads(text) if text else None

    def put_file(self, rel, local):
        local = Path(local)
        if not self.local:
            self.api.upload_file(path_or_fileobj=local, path_in_repo=self._hf(rel), repo_id=OUTPUT_REPO,
                                 repo_type="model")
            return
        data, parts = local.read_bytes(), []
        for n, start in enumerate(range(0, max(1, len(data)), self.PART)):
            name = f"part-{n:04d}.b64"
            assert FS.put(f"{self.root}/{rel}/{name}", base64.b64encode(data[start:start + self.PART]).decode(),
                          overwrite=True), f"{rel}/{name}"
            parts.append(name)
        self.put_json(f"{rel}/manifest.json",
                      {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "parts": parts})

    def get_file(self, rel, local):
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        if not self.local:
            shutil.copy(hf_hub_download(OUTPUT_REPO, self._hf(rel), repo_type="model", token=TOKEN), local)
            return local
        manifest = self.get_json(f"{rel}/manifest.json")
        if manifest is None:
            raise FileNotFoundError(f"{self.root}/{rel}")
        data = b"".join(base64.b64decode(self.get_text(f"{rel}/{name}")) for name in manifest["parts"])
        if len(data) != manifest["bytes"] or hashlib.sha256(data).hexdigest() != manifest["sha256"]:
            raise IOError(f"checksum mismatch for {self.root}/{rel}")
        local.write_bytes(data)
        return local

    def put_dir(self, rel, local_dir):
        local_dir = Path(local_dir)
        if not self.local:
            self.api.upload_folder(repo_id=OUTPUT_REPO, repo_type="model", folder_path=local_dir,
                                   path_in_repo=self._hf(rel), commit_message=f"{RUN_NAME}: {rel}")
            return
        files = sorted(str(p.relative_to(local_dir)) for p in local_dir.rglob("*") if p.is_file())
        for name in files:
            self.put_file(f"{rel}/{name}", local_dir / name)
        self.put_json(f"{rel}/files.json", files)   # written last: a folder without it is incomplete

    def get_dir(self, rel, local_dir):
        if not self.local:
            root = Path(snapshot_download(OUTPUT_REPO, repo_type="model", token=TOKEN,
                                          allow_patterns=[f"{self._hf(rel)}/**"]))
            shutil.copytree(root / self._hf(rel), local_dir, dirs_exist_ok=True)
            return Path(local_dir)
        files = self.get_json(f"{rel}/files.json")
        if files is None:
            raise FileNotFoundError(f"{self.root}/{rel} (no files.json: missing or incomplete)")
        for name in files:
            self.get_file(f"{rel}/{name}", Path(local_dir) / name)
        return Path(local_dir)

    def delete(self, rel=""):
        try:
            if self.local:
                FS.rm(f"{self.root}/{rel}".rstrip("/"), recurse=True)
            else:
                self.api.delete_folder(repo_id=OUTPUT_REPO, repo_type="model", path_in_repo=self._hf(rel).rstrip("/"),
                                       commit_message=f"{RUN_NAME}: delete {rel}")
        except Exception as exc:
            print(json.dumps({"event": "delete_warning", "path": rel, "error": str(exc)}))

    def checkpoint_steps(self):
        try:
            if self.local:
                names = [f.name.rstrip("/") for f in FS.ls(f"{self.root}/checkpoints")]
            else:
                prefix = f"{self.root}/checkpoints/"
                names = [p[len(prefix):].split("/")[0] for p in self.api.list_repo_files(OUTPUT_REPO, repo_type="model")
                         if p.startswith(prefix)]
        except Exception:
            return set()
        return {int(m.group(1)) for n in names if (m := re.fullmatch(r"step-(\d{8})", n))}


STORE_ = None   # the run's Store, created in main()


def upload_json(api, path, value):
    api.upload_file(path_or_fileobj=io.BytesIO(json.dumps(value, indent=2, default=str).encode()),
                    path_in_repo=path, repo_id=OUTPUT_REPO, repo_type="model")


def remote_best_step():
    best = STORE_.get_json("best.json")
    return int(best["step"]) if best else None


def prune_remote_checkpoints(api, latest_step, best_step):
    keep = {int(latest_step)} | ({int(best_step)} if best_step is not None else set())
    for old_step in sorted(STORE_.checkpoint_steps() - keep):
        STORE_.delete(f"checkpoints/step-{old_step:08d}")
        print(json.dumps({"event": "checkpoint_pruned", "step": old_step}))


def publish_public_best(api, model_dir, step, best, metrics):
    if not PUBLIC_OUTPUT_REPO:
        return
    try:
        api.create_repo(PUBLIC_OUTPUT_REPO, repo_type="model", private=False, exist_ok=True)
        api.upload_folder(repo_id=PUBLIC_OUTPUT_REPO, repo_type="model", folder_path=model_dir,
                          path_in_repo="", commit_message=f"Publish selected best checkpoint at step {step}")
        api.upload_file(
            path_or_fileobj=io.BytesIO(json.dumps({
                "source_private_repo": OUTPUT_REPO, "run_name": RUN_NAME, "step": step,
                "best_selection_score": best, "metrics": metrics,
            }, indent=2, default=str).encode()),
            path_in_repo="training_summary.json", repo_id=PUBLIC_OUTPUT_REPO, repo_type="model",
            commit_message=f"Update evaluation summary for step {step}")
        config_path = RUN_DIR / "config.json"
        if config_path.exists():
            api.upload_file(path_or_fileobj=config_path, path_in_repo="training_config.json",
                            repo_id=PUBLIC_OUTPUT_REPO, repo_type="model")
        card = f"""---
library_name: transformers
base_model: {BASE_MODEL}
license: apache-2.0
pipeline_tag: feature-extraction
tags:
- siglip
- image-text-retrieval
- home-decor
---

# Home Decor SigLIP 2

Experimental SigLIP 2 checkpoint fine-tuned for home-decor image/text retrieval.
This repository contains the selected best inference model only.

- Base model: `{BASE_MODEL}`
- Training run: `{RUN_NAME}`
- Selected step: `{step}`
- Selection score: `{best:.6f}`
- Effective contrastive batch: `{EFF_BATCH}` (GradCache)

See `training_summary.json` for validation metrics and `training_config.json` for
the training configuration. Evaluate on your own held-out data before production use.
"""
        api.upload_file(path_or_fileobj=io.BytesIO(card.encode()), path_in_repo="README.md",
                        repo_id=PUBLIC_OUTPUT_REPO, repo_type="model",
                        commit_message=f"Update model card for step {step}")
        print(json.dumps({"event": "public_best_published", "repo": PUBLIC_OUTPUT_REPO, "step": step}))
    except Exception as exc:
        print(json.dumps({"event": "public_publish_warning", "error": str(exc)}))


def remote_checkpoint():
    """(local checkpoint folder with model/, last.json content, resume-state path or None)."""
    last = STORE_.get_json("last.json")
    if not last:
        return None, None, None
    local = RUN_DIR / "restore" / f"step-{int(last['step']):08d}"
    STORE_.get_dir(f"checkpoints/step-{int(last['step']):08d}/model", local / "model")
    state = None
    if not STORE_.local:   # Local runs keep weights only
        try:
            state = STORE_.get_file("resume/training_state.pt", local / "training_state.pt")
        except Exception:
            state = None
    return local, last, state


def export_model(model, processor, dest, bf16=False):
    dest.mkdir(parents=True, exist_ok=True)
    if USE_LORA:
        model.save_pretrained(dest / "lora_adapter")
        model.merge_adapter()
        try:
            model.get_base_model().save_pretrained(dest, safe_serialization=True)
        finally:
            model.unmerge_adapter()
    elif bf16:   # half the size on disk; loaders upcast to float32
        state = {k: (v.detach().to(torch.bfloat16) if v.is_floating_point() else v)
                 for k, v in model.state_dict().items()}
        model.save_pretrained(dest, safe_serialization=True, state_dict=state)
    else:
        model.save_pretrained(dest, safe_serialization=True)
    processor.save_pretrained(dest)
    gc.collect(); torch.cuda.empty_cache()


def selection_score(metrics, baseline):
    if SELECTION == "multi":
        # Equal-weight mean over data types, all from the validation split (no STL).
        for key in ("query_scenes", "query_crops", "query_abo"):
            if not metrics.get(key):
                raise RuntimeError(f"validation metric {key} is missing: the validation split needs rc_queries "
                                   "(recaption.py SPLIT='validation', attached by run_training.py)")
        values = {
            "df_crop_R@1": metrics["df_crop_to_render"]["R@1"],
            "scene_query_R@10": metrics["query_scenes"]["R@10"],
            "crop_query_R@10": metrics["query_crops"]["R@10"],
            "abo_query_R@1": metrics["query_abo"]["R@1"],
            "category_nDCG@10": metrics["attributes"]["category"]["nDCG@10"],
        }
        gates = {
            "df_R@10": metrics["df_crop_to_render"]["R@10"] >= 0.95 * baseline["df_crop_to_render"]["R@10"],
            "category_nDCG@10": values["category_nDCG@10"] >= 0.98 * baseline["attributes"]["category"]["nDCG@10"],
        }
        return float(sum(values.values()) / len(values)), values, gates, all(gates.values())
    # R@1 now carries real weight: top-10 recall was already passing while the
    # first-result accuracy that users actually feel was not.
    values = {
        "stl_R@1": metrics["stl_crop_to_product"]["R@1"],
        "stl_R@10": metrics["stl_crop_to_product"]["R@10"],
        "df_R@1": metrics["df_crop_to_render"]["R@1"],
        "df_R@10": metrics["df_crop_to_render"]["R@10"],
        "style_nDCG@10": metrics["attributes"]["style"]["nDCG@10"],
        "category_nDCG@10": metrics["attributes"]["category"]["nDCG@10"],
    }
    score = (
        0.35 * values["stl_R@1"]
        + 0.30 * values["stl_R@10"]
        + 0.15 * values["df_R@1"]
        + 0.10 * values["style_nDCG@10"]
        + 0.10 * values["category_nDCG@10"]
    )
    gates = {
        "df_R@10": values["df_R@10"] >= 0.95 * baseline["df_crop_to_render"]["R@10"],
        "category_nDCG@10": values["category_nDCG@10"] >= 0.98 * baseline["attributes"]["category"]["nDCG@10"],
    }
    return float(score), values, gates, all(gates.values())


def save_checkpoint(model, processor, optimizer, scheduler, scaler, step, best, api, metrics, is_best, final=False):
    if STORE_.local:
        save_checkpoint_local(model, processor, step, best, metrics, is_best, final)
        return
    checkpoint = RUN_DIR / "checkpoints" / f"step-{step:08d}"
    model_dir = checkpoint / "model"
    export_model(model, processor, model_dir)
    (checkpoint / "metrics.json").write_text(json.dumps(metrics, indent=2))
    training_state = {
        "step": step, "best_selection_score": best,
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(), "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }
    if USE_LORA:
        training_state["trainable_model_state"] = {
            name: param.detach().cpu() for name, param in model.named_parameters() if param.requires_grad
        }
    resume_state = RUN_DIR / "resume" / "training_state.pt"
    resume_state.parent.mkdir(parents=True, exist_ok=True)
    torch.save(training_state, resume_state)
    api.upload_folder(repo_id=OUTPUT_REPO, repo_type="model", folder_path=checkpoint,
                      path_in_repo=f"runs/{RUN_NAME}/checkpoints/{checkpoint.name}",
                      commit_message=f"{RUN_NAME}: checkpoint {step}")
    api.upload_file(path_or_fileobj=resume_state,
                    path_in_repo=f"runs/{RUN_NAME}/resume/training_state.pt",
                    repo_id=OUTPUT_REPO, repo_type="model",
                    commit_message=f"{RUN_NAME}: resumable state at {step}")
    upload_json(api, f"runs/{RUN_NAME}/last.json",
                {"run_name": RUN_NAME, "step": step, "best_step": step if is_best else remote_best_step(),
                 "best_selection_score": best})
    if is_best:
        upload_json(api, f"runs/{RUN_NAME}/best.json", {"run_name": RUN_NAME, "step": step,
                                                         "best_selection_score": best, "metrics": metrics})
        publish_public_best(api, model_dir, step, best, metrics)
    best_step = step if is_best else remote_best_step()
    prune_remote_checkpoints(api, latest_step=step, best_step=best_step)
    for old in sorted((RUN_DIR / "checkpoints").glob("step-*"))[:-1]:
        shutil.rmtree(old, ignore_errors=True)
    for name in ("log.jsonl", "config.json"):
        if (RUN_DIR / name).exists():
            api.upload_file(path_or_fileobj=RUN_DIR / name, path_in_repo=f"runs/{RUN_NAME}/{name}",
                            repo_id=OUTPUT_REPO, repo_type="model")


def save_checkpoint_local(model, processor, step, best, metrics, is_best, final):
    """Local runs: metrics at every evaluation; bf16 weights only for a new best and at the end (about 2 min each at
    ~9 MB/s). No optimizer state: an interrupted run restarts from the last saved weights."""
    STORE_.put_json(f"metrics/step-{step:08d}.json", metrics)
    if is_best or final:
        local = RUN_DIR / "checkpoints" / f"step-{step:08d}" / "model"
        export_model(model, processor, local, bf16=True)
        STORE_.put_dir(f"checkpoints/step-{step:08d}/model", local)
        best_step = step if is_best else remote_best_step()
        STORE_.put_json("last.json", {"run_name": RUN_NAME, "step": step, "best_step": best_step,
                                      "best_selection_score": best})
        if is_best:
            STORE_.put_json("best.json", {"run_name": RUN_NAME, "step": step, "best_selection_score": best,
                                          "metrics": metrics})
        prune_remote_checkpoints(None, latest_step=step, best_step=best_step)
        shutil.rmtree(local.parent, ignore_errors=True)
        print(json.dumps({"event": "checkpoint_saved", "step": step, "best": is_best, "final": final,
                          "where": f"{STORE_.root}/checkpoints/step-{step:08d}/model"}), flush=True)
    for name in ("log.jsonl", "config.json"):
        if (RUN_DIR / name).exists():
            STORE_.put_text(name, (RUN_DIR / name).read_text())


def audit_datasets(train_sets, validation_sets):
    required = {
        "abo": {"image", "group_id"},
        "deepfurniture_crops": {"image", "furniture_id"},
        "deepfurniture_scenes": {"image"},
        "deepfurniture_products": {"image", "furniture_id"},
    }
    report = {"splits": {}, "pairing": {}, "blockers": [], "warnings": []}
    id_names = ("group_id", "furniture_id", "identity_id", "product_id", "scene_id", "image_id", "id")

    def groups(dataset, source):
        columns = [name for name in id_names if name in dataset.column_names]
        metadata = dataset.select_columns(columns)
        return [identity(metadata[n], source, n) for n in range(len(metadata))]

    for source in CONFIGS:
        train, validation = train_sets[source], validation_sets[source]
        missing = sorted(required.get(source, {"image", "group_id"}) - set(train.column_names))
        if missing:
            report["blockers"].append(f"{source}: missing required columns {missing}")
        train_groups, validation_groups = groups(train, source), groups(validation, source)
        overlap = set(train_groups) & set(validation_groups)
        if overlap:
            report["blockers"].append(
                f"{source}: {len(overlap)} product/group identities occur in train and validation")
        coverage, field_quality = {}, {}
        for field in FIELD_WEIGHTS:
            if field in train.column_names:
                normalized = [field_text(field, value) for value in train[field]]
                present_values = [value for value in normalized if value]
                coverage[field] = round(len(present_values) / max(1, len(train)), 4)
                frequencies = Counter(present_values)
                by_group = defaultdict(set)
                for group, value in zip(train_groups, normalized):
                    if value:
                        by_group[group].add(value)
                conflicts = sum(len(values) > 1 for values in by_group.values())
                field_quality[field] = {
                    "unique_values": len(frequencies),
                    "most_common": frequencies.most_common(5),
                    "dominant_value_share": round(
                        (frequencies.most_common(1)[0][1] / len(present_values)) if present_values else 0.0, 4),
                    "groups_with_conflicting_values": conflicts,
                    "conflicting_group_share": round(conflicts / max(1, len(by_group)), 4),
                }
                if source == "abo" and field in SHORT_FIELDS and conflicts / max(1, len(by_group)) > 0.10:
                    report["warnings"].append(
                        f"{source}.{field}: {conflicts / max(1, len(by_group)):.1%} of labeled groups "
                        "have conflicting values across views")
        counts = Counter(train_groups)
        repeats = np.asarray(list(counts.values()) or [0])
        report["splits"][source] = {
            "train_rows": len(train), "validation_rows": len(validation),
            "train_groups": len(counts), "validation_groups": len(set(validation_groups)),
            "cross_split_group_overlap": len(overlap), "field_coverage": coverage,
            "field_quality": field_quality,
            "views_per_group_p50": float(np.percentile(repeats, 50)),
            "views_per_group_p95": float(np.percentile(repeats, 95)),
            "views_per_group_max": int(repeats.max()),
        }

        decode_errors = []
        sample = list(range(len(train)))
        random.Random(SEED + len(source)).shuffle(sample)
        for index in sample[:min(32, len(sample))]:
            try:
                image = as_pil(image_value(train[index]))
                image.load()
                if min(image.size) < 16:
                    decode_errors.append(f"row {index}: suspicious size {image.size}")
            except Exception as exc:
                decode_errors.append(f"row {index}: {type(exc).__name__}: {exc}")
        report["splits"][source]["sample_decode_errors"] = decode_errors
        if decode_errors:
            report["blockers"].append(f"{source}: sampled image decode errors: {decode_errors[:3]}")

    products = {str(value) for value in train_sets["deepfurniture_products"]["furniture_id"]}
    crop_ids = [str(value) for value in train_sets["deepfurniture_crops"]["furniture_id"]]
    crop_coverage = sum(value in products for value in crop_ids) / max(1, len(crop_ids))
    abo_groups = Counter(str(value) for value in train_sets["abo"]["group_id"])
    abo_multi = sum(abo_groups[str(value)] > 1 for value in train_sets["abo"]["group_id"]) / max(
        1, len(train_sets["abo"]))
    report["pairing"] = {
        "crop_rows_with_product_render": round(crop_coverage, 4),
        "abo_rows_with_another_view": round(abo_multi, 4),
    }
    if crop_coverage < 0.95:
        report["warnings"].append(f"Only {crop_coverage:.1%} of crop rows have a product render partner")
    if abo_multi < 0.25:
        report["warnings"].append(f"Only {abo_multi:.1%} of ABO rows have another product view")
    return report


def data_audit_preview(train_sets, path, per_source=6):
    render_rows = {}
    for index, value in enumerate(train_sets["deepfurniture_products"]["furniture_id"]):
        render_rows.setdefault(str(value), index)
    abo_rows = defaultdict(list)
    for index, value in enumerate(train_sets["abo"]["group_id"]):
        abo_rows[str(value)].append(index)

    samples = []
    for source in SOURCES:
        candidates = list(range(len(train_sets[source])))
        random.Random(SEED + len(source)).shuffle(candidates)
        for index in candidates[:per_source]:
            row = train_sets[source][index]
            partner, kind = None, ""
            if source == "deepfurniture_crops":
                match = render_rows.get(product_id(row))
                if match is not None:
                    partner = image_value(train_sets["deepfurniture_products"][match])
                    kind = "crop -> render"
            elif source == "abo":
                alternatives = [value for value in abo_rows[str(row["group_id"])] if value != index]
                if alternatives:
                    partner = image_value(train_sets["abo"][alternatives[0]])
                    kind = "ABO view -> view"
            samples.append((source, index, row, partner, kind))

    image_size, row_height, width = 224, 250, 1180
    sheet = Image.new("RGB", (width, row_height * len(samples)), "white")
    draw = ImageDraw.Draw(sheet)
    for number, (source, index, row, partner, kind) in enumerate(samples):
        top = number * row_height
        raw = as_pil(image_value(row))
        anchor = raw.copy(); anchor.thumbnail((image_size, image_size))
        sheet.paste(anchor, ((image_size - anchor.width) // 2, top + (image_size - anchor.height) // 2))
        # Third column shows what the model actually sees after augmentation.
        augmented = domain_augment(raw.copy(), random.Random(SEED + index), AUG_STRENGTH,
                                   heavy=(source == "abo"))
        augmented.thumbnail((image_size, image_size))
        if partner is not None:
            other = as_pil(partner); other.thumbnail((image_size, image_size))
            sheet.paste(other, (image_size + 12 + (image_size - other.width) // 2,
                                top + (image_size - other.height) // 2))
        sheet.paste(augmented, (2 * (image_size + 12) + (image_size - augmented.width) // 2,
                                top + (image_size - augmented.height) // 2))
        fields = []
        for field in ("title", "category", "style", "vlm_style", "material", "color", "vlm_caption"):
            if field in row and field_text(field, row.get(field)):
                fields.append(f"{field}: {field_text(field, row.get(field))[:70]}")
        label = f"{source} row={index}  {kind or 'no partner'}  [3rd = augmented anchor]\n" + "\n".join(fields[:7])
        draw.multiline_text((3 * image_size + 44, top + 8), label, fill="black", spacing=4)
        draw.line((0, top + row_height - 1, width, top + row_height - 1), fill="#cccccc")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)

# ------------------------------------------------------------------ main


def dry_run_subsets(sets, n=64):
    """A few rows per config; products limited to those the sampled crops point to (so crop->render works)."""
    out = {c: select(x, n, SEED) for c, x in sets.items() if c != "deepfurniture_products"}
    if "deepfurniture_products" in sets:
        wanted = {str(x) for x in out["deepfurniture_crops"]["furniture_id"]} if "deepfurniture_crops" in out else set()
        pids = [str(x) for x in sets["deepfurniture_products"]["furniture_id"]]
        keep = [i for i, p in enumerate(pids) if p in wanted][:n] or list(range(min(n, len(pids))))
        out["deepfurniture_products"] = sets["deepfurniture_products"].select(keep)
    return out


def dry_run_checks(model, processor, mixed, validation_sets, device, dtype):
    """CPU plumbing check: training texts per source, validation + selection score, checkpoint save to the
    store, reload, compare one weight tensor, delete the dry-run outputs. Raises if anything is off."""
    problems = []
    # 1. What each source really trains on (after DROP_FIELDS and re-captions)
    items = [mixed[(mixed.global_index(source, local), 0)]
             for source in SOURCES for local in range(min(3, len(mixed.datasets[source])))]
    batch = Collator(processor)(items)
    fields = {s: sorted({f for x in items if x["source"] == s for f, v in x["fields"].items() if v}) for s in SOURCES}
    print("DRY RUN | text fields used per source:\n" + json.dumps(fields, indent=1), flush=True)
    print("DRY RUN | example texts:\n" + json.dumps(
        [{"source": x["source"], "fields": {k: v for k, v in x["fields"].items() if v}} for x in items[::3]],
        indent=1)[:4000], flush=True)
    for source in SOURCES:
        for f in DROP_FIELDS.get(source, set()) & set(fields.get(source, [])):
            problems.append(f"{source}: dropped field '{f}' is still used")
        columns = mixed.datasets[source].column_names
        if "rc_caption" in columns and source != "abo" and "rc_caption" not in fields[source]:
            problems.append(f"{source}: rc_caption column exists but is not used")
        if "rc_queries" in columns and "rc_query" not in fields[source]:
            problems.append(f"{source}: rc_queries column exists but rc_query is not used")
    print(f"DRY RUN | batch of {len(items)} images -> {int(batch['input_ids'].shape[0])} texts, partners: "
          f"{dict(Counter(x['partner_kind'] or 'none' for x in items))}", flush=True)

    # 2. Validation and checkpoint selection
    metrics = validate(model, processor, validation_sets, device, dtype, smoke=False)
    score, values, gates, eligible = selection_score(metrics, metrics)
    print("DRY RUN | validation:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                                for k, v in values.items()}), "| score", round(score, 4), flush=True)

    # 3. Save to the store, reload, compare
    if STORE_.local:
        t0 = time.time()
        save_checkpoint_local(model, processor, 1, score, metrics, is_best=True, final=True)
        saved = time.time() - t0
        local, last, _ = remote_checkpoint()
        from transformers import AutoModel
        reloaded = AutoModel.from_pretrained(local / "model", dtype=torch.float32).state_dict()
        name, original = next((k, v) for k, v in model.state_dict().items() if v.is_floating_point() and v.numel() > 1000)
        same = torch.equal(reloaded[name].cpu(), original.detach().cpu().to(torch.bfloat16).float())
        print(f"DRY RUN | checkpoint saved in {saved / 60:.1f} min, reloaded from {STORE_.root}; "
              f"tensor '{name}' identical after bf16 round trip: {same}", flush=True)
        if not same:
            problems.append("reloaded weights differ from the saved ones")
        STORE_.delete()   # dry-run outputs only
    else:
        print("DRY RUN | store is HF: checkpoint round trip skipped", flush=True)

    if problems:
        raise RuntimeError("DRY RUN FAILED:\n- " + "\n- ".join(problems))
    print("DRY RUN PASSED", flush=True)


def main():
    global TOKEN, STORE_
    from datasets import load_from_disk

    TOKEN = get_token()
    login(TOKEN)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUN_DIR / "log.jsonl"
    STORE_ = Store()
    api = getattr(STORE_, "api", None)   # only used by the old HF-only paths
    print(json.dumps({"event": "store", "store": STORE, "root": STORE_.root, "selection": SELECTION,
                      "dry_run": DRY_RUN, "drop_fields": {k: sorted(v) for k, v in DROP_FIELDS.items()}}), flush=True)
    # Built from the data folder by the script, with the leak-checked rows already removed from train.
    train_sets = {c: load_from_disk(str(LOCAL_DATA / c / "train")) for c in CONFIGS}
    validation_sets = {c: load_from_disk(str(LOCAL_DATA / c / "validation")) for c in CONFIGS}
    if DRY_RUN:
        train_sets, validation_sets = dry_run_subsets(train_sets), dry_run_subsets(validation_sets)
    raw_counts = {c: len(x) for c, x in train_sets.items()}
    audit = audit_datasets(train_sets, validation_sets)
    audit_path = RUN_DIR / "data_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2))
    audit_preview = RUN_DIR / "artifacts" / "data_audit_preview.jpg"
    data_audit_preview(train_sets, audit_preview)
    STORE_.put_text("data_audit.json", audit_path.read_text())
    STORE_.put_file("artifacts/data_audit_preview.jpg", audit_preview)
    print("DATA AUDIT\n" + json.dumps(audit, indent=2), flush=True)
    if audit["blockers"]:
        raise RuntimeError("Data audit failed: " + "; ".join(audit["blockers"]))
    if AUDIT_ONLY:
        print("Audit passed; HOME_DECOR_AUDIT_ONLY=1, so training was not started.")
        return

    from transformers import AutoModel, AutoProcessor

    random.seed(SEED); torch.manual_seed(SEED)
    if DRY_RUN:   # CPU plumbing check, no training
        device, dtype, scaler = torch.device("cpu"), torch.float32, None
        print("DRY RUN on CPU | store:", STORE_.root)
    else:
        assert torch.cuda.is_available(), "A GPU is required"
        torch.cuda.manual_seed_all(SEED)
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        major, _ = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=False)  # GradCache backward is manual; bf16/fp32 only
        if dtype == torch.float16:
            # fp16 without a scaler underflows; T4 users fall back to fp32 autocast off.
            dtype = torch.float32
        print("GPU:", torch.cuda.get_device_name(0), "| dtype:", dtype,
              "| eff_batch:", EFF_BATCH, "| chunk:", CHUNK, "| LoRA:", USE_LORA)

    if SMOKE:
        limits = {"abo": 4_096, "deepfurniture_crops": 4_096, "deepfurniture_scenes": 2_048,
                  "deepfurniture_products": 0}
        train_sets = {c: select(x, limits.get(c, 1_024), SEED + n) for n, (c, x) in enumerate(train_sets.items())}

    render_index = {}
    for n, pid in enumerate(train_sets["deepfurniture_products"]["furniture_id"]):
        render_index.setdefault(str(pid), n)
    crop_pids = [str(x) for x in train_sets["deepfurniture_crops"]["furniture_id"]]
    crop_coverage = sum(p in render_index for p in crop_pids) / max(1, len(crop_pids))

    group_rows = defaultdict(list)
    for n, g in enumerate(train_sets["abo"]["group_id"]):
        group_rows[g].append(n)
    abo_views = {n: [m for m in rows if m != n] for rows in group_rows.values() for n in rows if len(rows) > 1}
    abo_view_coverage = len(abo_views) / max(1, len(train_sets["abo"]))

    pix3d_views = {}
    if "pix3d" in train_sets:
        pix3d_rows = defaultdict(list)
        for n, g in enumerate(train_sets["pix3d"]["group_id"]):
            pix3d_rows[g].append(n)
        pix3d_views = {n: [m for m in rows if m != n] for rows in pix3d_rows.values() for n in rows if len(rows) > 1}

    mixed = MixedDataset(train_sets, render_index, abo_views, pix3d_views)
    groups, weights, masses, probabilities = source_state(train_sets)
    categories = category_labels(train_sets) if HARD_NEG_FRACTION > 0 else None
    steps_per_epoch = max(1, math.ceil(sum(masses.values()) / EFF_BATCH))
    total_steps = 40 if SMOKE else (STEPS_OVERRIDE or max(1, int(round(EPOCHS * steps_per_epoch))))
    eval_every = total_steps if SMOKE else EVAL_EVERY
    warmup = max(1, round(total_steps * WARMUP_FRACTION))

    checkpoint, last, resume_path = remote_checkpoint()
    if last and not log_path.exists():
        # Continue the run's log instead of starting a new file that would overwrite it on upload.
        previous = STORE_.get_text("log.jsonl")
        if previous:
            log_path.write_text(previous)
        else:
            print(json.dumps({"event": "log_restore_warning", "error": "no log.jsonl in the store"}))
    source = checkpoint / "model" if checkpoint else BASE_MODEL
    processor = AutoProcessor.from_pretrained(source, token=TOKEN)
    model_source = BASE_MODEL if USE_LORA else source
    # float32 master weights even when the checkpoint was saved in bf16
    model = AutoModel.from_pretrained(model_source, token=TOKEN, dtype=torch.float32).to(device)
    if GRAD_CKPT:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    teacher = None
    if USE_LORA:
        from peft import LoraConfig, PeftModel, get_peft_model
        adapter = checkpoint / "model" / "lora_adapter" if checkpoint else None
        if adapter and adapter.exists():
            model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(
                r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                target_modules=LORA_TARGETS, bias="none"))
        for name, param in model.named_parameters():
            if "logit_scale" in name or "logit_bias" in name:
                param.requires_grad = True
        model.print_trainable_parameters()
    elif DISTILL_WEIGHT and not DRY_RUN:
        teacher = AutoModel.from_pretrained(BASE_MODEL, token=TOKEN, torch_dtype=dtype).to(device).eval()
        teacher.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config = {"base_model": BASE_MODEL, "run_name": RUN_NAME,
              "private_output_repo": OUTPUT_REPO, "public_output_repo": PUBLIC_OUTPUT_REPO or None,
              "smoke": SMOKE, "use_lora": USE_LORA,
              "gpu": torch.cuda.get_device_name(0), "dtype": str(dtype),
              "effective_batch": EFF_BATCH, "gradcache_chunk": CHUNK, "gradient_checkpointing": GRAD_CKPT,
              "epochs": EPOCHS, "total_steps": total_steps, "steps_per_epoch": steps_per_epoch, "lr": LR,
              "trainable_params": trainable, "raw_counts": raw_counts,
              "loaded_counts": {c: len(x) for c, x in train_sets.items()},
              "sampling_masses": masses, "source_probabilities": probabilities,
              "crop_render_coverage": crop_coverage, "abo_view_coverage": abo_view_coverage,
              "extra_sources": list(EXTRA), "local_data": str(LOCAL_DATA),
              "pix3d_view_coverage": (len(pix3d_views) / max(1, len(train_sets["pix3d"]))) if "pix3d" in train_sets else None,
              "field_weights": FIELD_WEIGHTS, "field_sample_k": FIELD_SAMPLE_K,
              "drop_fields": {k: sorted(v) for k, v in DROP_FIELDS.items()},
              "augmentation_strength": AUG_STRENGTH,
              "hard_negative_fraction": HARD_NEG_FRACTION, "hard_negative_categories": HARD_NEG_CATEGORIES,
              "image_pair_weight": IMAGE_PAIR_WEIGHT, "distill_weight": DISTILL_WEIGHT,
              "pair_loss_type": PAIR_LOSS, "infonce_tau": INFONCE_TAU, "stop_at_step": STOP_AT_STEP,
              "eval_every": eval_every, "stl_validation_queries": STL_VAL_QUERIES,
              "stl_validation_gallery": STL_VAL_GALLERY or "full",
              "max_hours": MAX_HOURS, "num_workers": NUM_WORKERS,
              "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "targets": LORA_TARGETS} if USE_LORA else None,
              "resumed_from": str(checkpoint) if checkpoint else None}
    (RUN_DIR / "config.json").write_text(json.dumps(config, indent=2))
    log(log_path, event="configuration", **config)
    if DRY_RUN:
        dry_run_checks(model, processor, mixed, validation_sets, device, dtype)
        return

    optimizer = build_optimizer(model)
    start = int(last["step"]) if last else 0
    best = float(last.get("best_selection_score", -1.0)) if last else -1.0
    rewarm_from = start

    def schedule(step):
        if step < warmup:
            factor = (step + 1) / warmup
        else:
            progress = (step - warmup) / max(1, total_steps - warmup)
            factor = .5 * (1 + math.cos(math.pi * min(1, progress)))
        if rewarm_from > 0:
            factor *= min(1.0, max(0.0, (step - rewarm_from + 1) / RESUME_REWARM_STEPS))
        return factor

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    if resume_path:
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if int(state["step"]) != start:
            raise RuntimeError(f"Resume state step {state['step']} does not match last checkpoint step {start}")
        if USE_LORA:
            parameters = dict(model.named_parameters())
            missing = sorted(set(state.get("trainable_model_state", {})) - set(parameters))
            if missing:
                raise RuntimeError(f"LoRA resume has unknown parameter names: {missing[:5]}")
            with torch.no_grad():
                for name, value in state.get("trainable_model_state", {}).items():
                    parameters[name].copy_(value.to(device=parameters[name].device, dtype=parameters[name].dtype))
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        best = float(state.get("best_selection_score", best))
        rewarm_from = 0
        log(log_path, event="resume_exact", step=start, best_selection_score=best)
    elif start:
        for _ in range(start):
            scheduler.step()
        log(log_path, event="resume_weights_only", step=start, best_selection_score=best,
            warning="optimizer/scheduler/RNG state unavailable")
    if start >= total_steps:
        print("already complete"); return

    dump_path = RUN_DIR / "artifacts" / "stl_top10_dump.jsonl"
    if start == 0:
        baseline = validate(model, processor, validation_sets, device, dtype, SMOKE, dump_path)
        log(log_path, event="validation_baseline", step=0, metrics=baseline)
        STORE_.put_json("baseline_step0.json", baseline)
        # A checkpoint must beat the untrained model to count as best (v1b-full picked a worse one).
        best = selection_score(baseline, baseline)[0]
        log(log_path, event="selection_floor", best_selection_score=best)
        if SELECTION == "stl":
            # Same positions stl_eval samples (random.sample depends only on the length and the seed), saved so
            # the final STL test can be reported on the queries that were NOT used to pick the checkpoint.
            n_queries = sum(1 for x in (STL_ROOT / "queries.jsonl").read_text().splitlines() if x)
            picked = sorted(random.Random(SEED).sample(range(n_queries), min(64 if SMOKE else STL_VAL_QUERIES, n_queries)))
            STORE_.put_json("stl_selection_queries.json", {"seed": SEED, "positions_in_queries_jsonl": picked})
            if dump_path.exists():
                STORE_.put_file("artifacts/stl_top10_base.jsonl", dump_path)
    else:
        baseline = STORE_.get_json("baseline_step0.json")
        if baseline is None:
            raise RuntimeError("resuming but baseline_step0.json is missing from the store")

    sampler = UniqueBatchSampler(mixed, groups, weights, probabilities, start, total_steps, categories)
    loader = torch.utils.data.DataLoader(mixed, batch_sampler=sampler, collate_fn=Collator(processor),
                                         num_workers=NUM_WORKERS, pin_memory=True,
                                         persistent_workers=True, prefetch_factor=2)
    optimizer.zero_grad(set_to_none=True); model.train()
    recent, all_losses, step, t_last = [], [], start, time.time()
    for step, batch in enumerate(loader, start=start + 1):
        if step == start + 1:
            preview = RUN_DIR / "artifacts" / "batch_preview.jpg"; batch_preview(batch, preview)
            STORE_.put_file("artifacts/batch_preview.jpg", preview)
            print("BATCH PAIRING PREVIEW\n" + json.dumps(batch["metadata"][:4], indent=2))

        loss_value, parts = gradcache_step(model, teacher, batch, device, dtype, CHUNK)
        grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        if not torch.isfinite(grad):
            log(log_path, event="skipped_step", step=step, reason="non-finite gradient norm")
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.step(); scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        all_losses.append(loss_value); recent = (recent + [loss_value])[-50:]
        if step == start + 1 or step % 5 == 0:
            now = time.time()
            elapsed = (now - START_TIME) / 3600
            per_step = (now - t_last) / (1 if step == start + 1 else 5)
            log(log_path, event="train", step=step, of=total_steps, loss=loss_value,
                loss_mean_50=sum(recent) / len(recent), losses=parts, lr=scheduler.get_last_lr()[0],
                grad_norm=float(grad), logit_scale=float(model.logit_scale.exp()),
                sec_per_step=round(per_step, 2),
                projected_total_hours=round(elapsed + per_step * (total_steps - step) / 3600, 2),
                peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1))
            t_last = now

        out_of_time = (time.time() - START_TIME) / 3600 > MAX_HOURS
        stop_now = bool(STOP_AT_STEP and step >= STOP_AT_STEP)
        if step % eval_every == 0 or step == total_steps or out_of_time or stop_now:
            metrics = validate(model, processor, validation_sets, device, dtype, SMOKE, dump_path)
            score, components, gates, eligible = selection_score(metrics, baseline)
            is_best = eligible and score > best
            if is_best:
                best = score
            metrics["checkpoint_selection"] = {
                "score": score, "components": components, "gates": gates, "eligible": eligible,
                "baseline": selection_score(baseline, baseline)[1],
            }
            log(log_path, event="validation", step=step, metrics=metrics, is_best=is_best,
                best_selection_score=best)
            save_checkpoint(model, processor, optimizer, scheduler, scaler, step, best, api, metrics, is_best,
                            final=(step == total_steps or out_of_time or stop_now))
            if is_best and SELECTION == "stl" and dump_path.exists():
                STORE_.put_file("artifacts/stl_top10_best.jsonl", dump_path)
        if out_of_time:
            log(log_path, event="time_limit", step=step, note="saved; run again to resume")
            break
        if stop_now:
            log(log_path, event="stopped_early", step=step,
                note="partial run; resume by unsetting STOP_AT_STEP")
            break

    result = {"run_name": RUN_NAME, "step": step, "of": total_steps, "best_selection_score": best}
    if SMOKE and all_losses:
        window = min(10, max(1, len(all_losses) // 2))
        result["first_loss_mean"] = sum(all_losses[:window]) / window
        result["last_loss_mean"] = sum(all_losses[-window:]) / window
        result["smoke_pass"] = result["last_loss_mean"] < result["first_loss_mean"]
    log(log_path, event="done", **result)
    STORE_.put_text("log.jsonl", log_path.read_text())
    print(result)


if __name__ == "__main__":
    main()
