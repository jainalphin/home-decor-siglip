---
license: cc-by-nc-4.0
base_model: google/siglip2-base-patch16-224
library_name: transformers
pipeline_tag: zero-shot-image-classification
tags:
  - siglip2
  - image-text-retrieval
  - furniture
  - interior-design
  - home-decor
---

# Home-decor SigLIP 2 (v3-clean-nostyle)

A fine-tune of [`google/siglip2-base-patch16-224`](https://huggingface.co/google/siglip2-base-patch16-224) for
**furniture and home-decor search**: real room photo → product, text → product, and text → room.

**Non-commercial use only** (CC BY-NC 4.0). Research / portfolio model.

Code (data cleaning, training, evaluation): [github.com/jainalphin/home-decor-siglip](https://github.com/jainalphin/home-decor-siglip).
Every evaluation table, including runs that did not help: [RESULTS.md](https://github.com/jainalphin/home-decor-siglip/blob/main/RESULTS.md).

## Results at a glance

All numbers are on held-out data that was never used for training or for choosing the checkpoint.

| Task | Base SigLIP 2 | This model |
|---|---|---|
| Real room photo → product (Shop the Look, 2,250 queries, 19k-product gallery), R@1 / R@10 | 0.076 / 0.364 | **0.164 / 0.508** |
| Real furniture photo → other photos of the same model (Pix3D, 6,795 queries), R@1 | 0.778 | **0.827** |
| Search query → room (DeepFurniture test rooms with no near-copy in training), R@10 | 0.281 | **0.391** |
| Search query → furniture crop (DeepFurniture test), R@10 | 0.361 | **0.440** |
| Search query → product photo (ABO test), R@1 | 0.620 | **0.680** |

95% bootstrap intervals for the "this model − base" differences exclude zero for the first four rows. For ABO only
R@10 was bootstrapped (+0.009, interval [0.000, 0.018]); both models are near the ceiling there.

## Known limits

The model is **slightly worse than base** on general room language, because its training text is shop titles and
magazine-style captions:

| Task | Base | This model |
|---|---|---|
| Human-written COCO captions of real home photos → photo, R@1 | 0.461 | 0.436 (−2.4) |
| Room type on real photos (Places365, 22 home room types), top-1 | 0.691 | 0.649 (−4.2) |

- The room-type loss is on rooms that are rare in the training data: basement, TV room, walk-in closet, home office,
  playroom. Living room, kitchen and kids room are better than base.
- Weakest product categories on real photos (Shop the Look R@1): curtain 0.028, sofa 0.069, throw pillow 0.107.
- Shop the Look "correct answers" are often similar products rather than the exact product, so its absolute
  scores understate the model; use them for comparisons.

## Intended use

- Search over furniture / decor catalogs and interior photos (text → image, image → image).
- Not for identifying people, not for commercial use.

## Usage

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

model_id = "AlphJain/home-decor-siglip2"
model = AutoModel.from_pretrained(model_id).eval()
processor = AutoProcessor.from_pretrained(model_id)

image = Image.open("room.jpg").convert("RGB")
texts = ["green velvet sofa", "scandinavian bedroom", "brass floor lamp"]   # lowercase, as in training
with torch.no_grad():
    img = model.get_image_features(**processor(images=image, return_tensors="pt"))
    txt = model.get_text_features(**processor(text=texts, padding="max_length", max_length=64,
                                              truncation=True, return_tensors="pt"))
img = torch.nn.functional.normalize(img, dim=-1)
txt = torch.nn.functional.normalize(txt, dim=-1)
print((img @ txt.T).softmax(-1))
```

## Training data

| Source | Rows used | What it is |
|---|---|---|
| Amazon Berkeley Objects (ABO) | 10,978 photos | Studio product photos with titles |
| DeepFurniture crops | 26,438 | Furniture cut out of rendered rooms, paired with the product render |
| DeepFurniture scenes | 15,727 | Rendered rooms |
| DeepFurniture products | 21,899 | Product renders (image-pair targets for the crops) |

Text per image: product titles (ABO), captions and attributes written by Qwen3-VL-32B, and a new caption plus three
short search queries per image written by gpt-4.1-mini.

Cleaning before training:
- Removed training images that are near-copies of any evaluation image (perceptual hash + SigLIP similarity).
- ABO: removed photos that belong to another listing, fabric-swatch listings, close-ups / back views / packaging
  (tagged by gpt-4.1-mini, checked by eye), and unreliable seller fields (keywords, colour, material, style).
- DeepFurniture: removed crops that are tiny, cut off, thin slivers or the wrong item; at most 5 crops per product.
- Removed the lowest 1% image-text matches and near-duplicate images (base SigLIP 2 scores).
- DeepFurniture's one-word style label and its template scene titles are not trained on (the captions name the
  style instead).

## Training

- Full fine-tune of both towers, 1,476 steps, batch 128 (GradCache), learning rate 5e-6, about 1.5 h on one A10G.
- Losses: SigLIP sigmoid loss per text field (short labels such as category treat identical texts as matches),
  InfoNCE on image pairs (DeepFurniture crop ↔ product render, two photos of the same ABO product), and 0.1
  distillation towards the base model's image features.
- Checkpoint chosen on a validation score over several data types (crop → product, scene / crop / ABO queries,
  category); Shop the Look was used only for the final test.

## What did not help

- Training on the uncleaned data (`v3-baseline`): lower Shop the Look R@1 (0.147 vs 0.164) and much worse text
  search (search query → room R@10 0.245 vs 0.334 on the full DeepFurniture test).
- Adding real photos with text only: Pix3D + COCO (`v3-real_photos`) and 13.5k cleaned Open Images photos with
  gpt-4.1-mini captions (`v3-clean-oi`). Open Images improved room types (+2.2) but lowered Shop the Look R@1 (−1.5)
  and COCO R@1 (−4.2). Real photo ↔ text pairs did not teach real photo ↔ catalog photo matching.

## Licenses and credits

- Base model: [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224).
- Amazon Berkeley Objects: CC BY 4.0 (Collins et al., 2022).
- DeepFurniture (Liu et al., 2019, [arXiv 1911.09299](https://arxiv.org/abs/1911.09299)), created and owned by COOHOM / Kujiale:
  AFL-3.0 on its [Hugging Face page](https://huggingface.co/datasets/byliu/DeepFurniture); the full dataset is shared under the
  owner's terms of use.
- Captions: Qwen3-VL-32B-Instruct and OpenAI gpt-4.1-mini outputs.
- Evaluation only (not trained on): Shop the Look (Pinterest), Pix3D, MS COCO, Places365.
- No training images are redistributed with this model.
