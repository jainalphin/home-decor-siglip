"""Build the private, held-out STL-Home evaluation set.

Run on Kaggle with internet access. The generated files are written to
``/kaggle/working/stl_eval`` on Kaggle and ``./stl_eval`` elsewhere.
"""

import io
import json
import random
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm.auto import tqdm


SEED = 0
N_EVAL_SCENES = 3000
MAX_QUERIES_PER_CAT = 250
N_DISTRACTORS = 8000
MIN_BOX_AREA = 0.01
MIN_CROP_SIDE = 48
MAX_SCENE_ASPECT = 2.2

OUT = (
    Path("/kaggle/working/stl_eval")
    if Path("/kaggle/working").exists()
    else Path("./stl_eval")
)

CAT_SHORT = {
    "Furniture|Chairs": "chair",
    "Home & Garden|Decor|Throw Pillows": "throw pillow",
    "Home & Garden|Lighting|Lighting Fixtures|Ceiling Light Fixtures": "ceiling light",
    "Home & Garden|Decor|Rugs": "rug",
    "Hardware|Plumbing|Plumbing Fixtures|Faucets": "faucet",
    "Home & Garden|Decor|Mirrors": "mirror",
    "Furniture|Chairs|Table & Bar Stools": "stool",
    "Home & Garden|Decor": "decor",
    "Furniture|Sofas": "sofa",
    "Home & Garden|Lighting|Lamps": "lamp",
    "Home & Garden|Decor|Window Treatments|Curtains & Drapes": "curtain",
}


def get_json(name):
    url = f"https://raw.githubusercontent.com/kang205/STL-Dataset/master/{name}"
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read().decode("utf-8")
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return [json.loads(line) for line in data.splitlines() if line.strip()]


def image_url(signature):
    return (
        "https://i.pinimg.com/400x/"
        f"{signature[0:2]}/{signature[2:4]}/{signature[4:6]}/{signature}.jpg"
    )


def fetch_to(signature, folder):
    path = OUT / folder / f"{signature}.jpg"
    if path.exists():
        return signature, True
    try:
        with urllib.request.urlopen(image_url(signature), timeout=20) as response:
            data = response.read()
        Image.open(io.BytesIO(data)).verify()
        path.write_bytes(data)
        return signature, True
    except Exception:
        return signature, False


def download(signatures, folder):
    downloaded = set()
    with ThreadPoolExecutor(32) as executor:
        jobs = executor.map(lambda signature: fetch_to(signature, folder), signatures)
        for signature, valid in tqdm(
            jobs, total=len(signatures), desc=folder
        ):
            if valid:
                downloaded.add(signature)
    return downloaded


def box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main():
    for directory in ["scenes", "products"]:
        (OUT / directory).mkdir(parents=True, exist_ok=True)

    pairs = get_json("home.json")
    categories = get_json("home-cat.json")
    print(f"[1] pairs: {len(pairs)}")

    random.seed(SEED)
    all_scenes = sorted({pair["scene"] for pair in pairs})
    eval_scenes = set(random.sample(all_scenes, N_EVAL_SCENES))
    eval_pairs = [pair for pair in pairs if pair["scene"] in eval_scenes]
    eval_products = {pair["product"] for pair in eval_pairs}
    other_products = sorted({pair["product"] for pair in pairs} - eval_products)
    distractors = set(
        random.sample(other_products, min(N_DISTRACTORS, len(other_products)))
    )
    print(
        f"    eval scenes: {len(eval_scenes)}  eval pairs: {len(eval_pairs)}  "
        f"eval products: {len(eval_products)}  distractors: {len(distractors)}"
    )

    eval_pairs = [
        pair for pair in eval_pairs if box_area(pair["bbox"]) >= MIN_BOX_AREA
    ]
    valid_scenes = download(
        sorted({pair["scene"] for pair in eval_pairs}), "scenes"
    )
    valid_products = download(
        sorted(eval_products | distractors), "products"
    )

    drops = Counter()
    queries = []
    for pair in eval_pairs:
        if (
            pair["scene"] not in valid_scenes
            or pair["product"] not in valid_products
        ):
            drops["missing_image"] += 1
            continue
        with Image.open(OUT / "scenes" / f"{pair['scene']}.jpg") as image:
            width, height = image.size
        if height / width > MAX_SCENE_ASPECT:
            drops["collage_aspect"] += 1
            continue
        left, top, right, bottom = pair["bbox"]
        if min((right - left) * width, (bottom - top) * height) < MIN_CROP_SIDE:
            drops["crop_too_small"] += 1
            continue
        raw_category = categories.get(pair["product"], "")
        queries.append(
            {
                "scene": pair["scene"],
                "product": pair["product"],
                "bbox": pair["bbox"],
                "category": CAT_SHORT.get(
                    raw_category, raw_category.split("|")[-1].lower()
                ),
                "category_raw": raw_category,
            }
        )

    queries_by_category = defaultdict(list)
    for query in queries:
        queries_by_category[query["category"]].append(query)
    queries = []
    for category, category_queries in queries_by_category.items():
        random.shuffle(category_queries)
        queries.extend(category_queries[:MAX_QUERIES_PER_CAT])
    for query_id, query in enumerate(queries):
        query["query_id"] = query_id

    scene_category_products = defaultdict(set)
    for pair in pairs:
        category = CAT_SHORT.get(categories.get(pair["product"], ""), "")
        if pair["scene"] in eval_scenes and pair["product"] in valid_products:
            scene_category_products[(pair["scene"], category)].add(pair["product"])
    for query in queries:
        relevant = scene_category_products[(query["scene"], query["category"])]
        query["relevant"] = sorted(relevant | {query["product"]})

    gallery = sorted((eval_products | distractors) & valid_products)
    print(f"[3] dropped: {dict(drops)}")
    print(f"    queries: {len(queries)}  gallery: {len(gallery)}")
    print(
        "    queries per category:",
        dict(Counter(query["category"] for query in queries).most_common()),
    )

    write_jsonl(OUT / "queries.jsonl", queries)
    write_jsonl(
        OUT / "gallery.jsonl",
        [
            {
                "product": signature,
                "category_raw": categories.get(signature, ""),
                "is_distractor": signature in distractors,
            }
            for signature in gallery
        ],
    )
    (OUT / "heldout_scenes.txt").write_text(
        "\n".join(sorted(eval_scenes)), encoding="utf-8"
    )
    (OUT / "heldout_products.txt").write_text(
        "\n".join(sorted(eval_products | distractors)), encoding="utf-8"
    )

    used = {query["scene"] for query in queries} | set(gallery)
    for directory in ["scenes", "products"]:
        for image_path in (OUT / directory).glob("*.jpg"):
            if image_path.stem not in used:
                image_path.unlink()
    print(f"[4] saved to {OUT}")

    random.seed(1)
    preview_queries = random.sample(queries, min(8, len(queries)))
    figure, axes = plt.subplots(
        2, len(preview_queries), figsize=(3 * len(preview_queries), 6.5)
    )
    if len(preview_queries) == 1:
        axes = np.array(axes).reshape(2, 1)
    for index, query in enumerate(preview_queries):
        scene = Image.open(OUT / "scenes" / f"{query['scene']}.jpg")
        width, height = scene.size
        left, top, right, bottom = query["bbox"]
        axes[0, index].imshow(scene)
        axes[0, index].add_patch(
            patches.Rectangle(
                (left * width, top * height),
                (right - left) * width,
                (bottom - top) * height,
                fill=False,
                color="lime",
                linewidth=2,
            )
        )
        axes[0, index].set_title(
            f"{query['category']} ({len(query['relevant'])} rel)", fontsize=8
        )
        axes[1, index].imshow(
            Image.open(OUT / "products" / f"{query['product']}.jpg")
        )
        for axis in axes[:, index]:
            axis.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
