# Home-decor SigLIP 2

Fine-tuning [SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-224) for furniture and home-decor search:
real room photo → product, text → product, text → room. This repo holds the data-cleaning, training and
evaluation scripts. No data or model weights are included.

**Result:** Shop the Look (real room photo → product, 2,250 held-out queries) R@1 0.076 → **0.164**, R@10
0.364 → **0.508**. Every evaluation table, including the runs that did not help: [RESULTS.md](RESULTS.md).
Model summary, known limits and credits: [MODEL_CARD.md](MODEL_CARD.md).

## How to run

Plain Python scripts, run from the repo root. Each one runs top to bottom; its settings (run ids, limits,
which models to compare) are constants in the first block of the file, so edit those before running.

```bash
pip install -r requirements.txt

export HF_TOKEN=hf_...            # Hugging Face (base dataset, eval sets, model checkpoints)
export OPENAI_API_KEY=sk-...      # only for the OpenAI caption / labelling scripts

python data/export_existing.py            # 1. copy the base dataset into decore/
python data/pix3d.py                      #    Pix3D + COCO (used by the leak check and the external eval)
python data/coco.py
python data/run_leak_check.py             # 2. leak check (GPU)
python data/apply_leak_rule.py            #    final leak rule -> exclude_v2.txt
python data/openai_label_jobs.py          # 3. photo tags (OpenAI)
python data/recaption.py                  #    new captions + queries (run with LIMIT = 40 first, then None)
python data/score_clean_dataset.py        #    image-text match + near-duplicates (GPU)
python data/build_clean_dataset.py        #    clean training split -> prints its run id
python finetuning/run_training.py         # 4. set CLEAN_RUN to that run id; DRY_RUN = True first, then False (GPU)
python eval/eval_stl_and_text.py          # 5. evaluation (GPU)
python eval/eval_external.py
python eval/eval_room_type.py
```

- All data is written under `./decore/` (`sources/`, `checks/`, `clean/`, `models/`, `evals/`). Each script
  prints a run id; the next step's settings take that run id.
- Review pages (photo grids to check by eye) are saved as HTML files in `./pages/`.
- [fsutil.py](fsutil.py) holds the small file helpers every script uses.
- GPU steps were run on one NVIDIA A10G (24 GB); `HOME_DECOR_CHUNK` in `run_training.py` lowers memory use.
- `data/vlm_captions_modal.py` runs on [Modal](https://modal.com) instead: `modal run data/vlm_captions_modal.py`.
- The `AlphJain/*` Hugging Face repos in the code are private; point them at your own copies. Run ids such as
  `20261108T155908Z` are from the original runs; replace them with yours.
- Keys are only read from environment variables; never write them into the files.

## Pipeline

**1. Source data** (`data/`)

| Script | What it does |
|---|---|
| `vlm_captions_modal.py` (+ `vocab.json`) | Qwen3-VL-32B captions and attributes (style, materials, colours, details) for every ABO and DeepFurniture product (Modal, vLLM) |
| `export_existing.py` | Copies the base dataset (ABO + DeepFurniture: products, crops, scenes) from Hugging Face into `decore/` |
| `pix3d.py`, `coco.py` | Real-photo sets (Pix3D furniture, COCO home rooms), used for the leak check and the external evaluation |

The script that first built the base Hugging Face dataset from ABO and DeepFurniture is not included.

**2. Leak check** (`data/`)

| Script | What it does |
|---|---|
| `leak_check.py` | Module: compares every training image with every evaluation image (pixel hash, dhash, base SigLIP 2 similarity) |
| `run_leak_check.py` | Runs it over all sources and writes an exclude list per source |
| `apply_leak_rule.py` | Re-applies a stricter rule to an existing run without re-embedding |

**3. Cleaning and new text** (`data/`)

| Script | What it does |
|---|---|
| `openai_label_jobs.py` | gpt-4.1-mini jobs: tags every ABO photo (product / detail / back / other item / other), judges Shop the Look labels, small checks |
| `score_clean_dataset.py` | Base SigLIP 2 image-text match score and near-duplicate detection |
| `recaption.py` | gpt-4.1-mini writes a caption and 3 search queries per DeepFurniture scene / crop and per ABO product; flags wrong-item crops |
| `build_clean_dataset.py` | Applies every rule above and writes the clean training split |

**4. Training** (`finetuning/`)

| Script | What it does |
|---|---|
| `train_v3.py` | Trainer module: GradCache contrastive training, per-field SigLIP loss, InfoNCE image pairs, distillation, multi-task checkpoint selection, checkpoints under `decore/models`, CPU dry run |
| `run_training.py` | Builds the local training data from `decore/` and runs `train_v3` (`DRY_RUN = True` first, on CPU) |

**5. Evaluation** (`eval/`, GPU; results in [RESULTS.md](RESULTS.md))

| Script | What it does |
|---|---|
| `eval_stl_and_text.py` | Shop the Look test (held-out queries, per category, bootstrap intervals) and text-search tests on the test split |
| `eval_external.py` | COCO human captions → photo, Pix3D photo → same furniture model |
| `eval_room_type.py` | Zero-shot room type on Places365 home rooms |
| `eval_scene_leak.py` | Checks whether test rooms also appear in training from another camera angle |

Scripts for the Open Images experiment (did not beat the final model) and the one-off data audit are not
included; their results are in [RESULTS.md](RESULTS.md).

## Data licenses

Amazon Berkeley Objects: CC BY 4.0. DeepFurniture: check its license. Open Images (experiment only, not in the final model): annotations CC BY 4.0,
images listed as CC BY 2.0. Shop the Look, Pix3D, COCO and Places365 were used for evaluation only. Captions
were generated with Qwen3-VL-32B-Instruct and OpenAI gpt-4.1-mini. The model itself is released as
non-commercial (CC BY-NC 4.0).
