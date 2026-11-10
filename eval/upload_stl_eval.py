"""Upload a built STL-Home evaluation set to a private HF dataset repo."""

import os
import tarfile
from pathlib import Path

from huggingface_hub import HfApi, login


REPO = "AlphJain/stl-home-eval"
SOURCE = (
    Path("/kaggle/working/stl_eval")
    if Path("/kaggle/working").exists()
    else Path("./stl_eval")
)
ARCHIVE_PATH = Path("/tmp/stl_eval_images.tar")
METADATA_FILES = [
    "queries.jsonl",
    "gallery.jsonl",
    "heldout_scenes.txt",
    "heldout_products.txt",
]

README = """---
private: true
---

# STL-Home evaluation set

Private held-out real-photo evaluation set built from Pinterest Shop The Look
(STL-Home). Images belong to Pinterest and its partners; no license is stated.
Personal research use only. Do not make this repository public.

Kang et al., "Complete the Look: Scene-based Complementary Product
Recommendation", CVPR 2019.

- `queries.jsonl`: 2,750 crop queries across 11 categories
- `gallery.jsonl`: evaluation products and distractors
- `heldout_scenes.txt` / `heldout_products.txt`: exclude from STL training
- `images.tar`: `scenes/` and `products/`
"""


def get_token():
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as error:
        raise RuntimeError(
            "Set HF_TOKEN in the environment or Kaggle Secrets"
        ) from error


def validate_source():
    required = [SOURCE / filename for filename in METADATA_FILES]
    required.extend([SOURCE / "scenes", SOURCE / "products"])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete STL evaluation set: {missing}")


def main():
    validate_source()
    token = get_token()
    login(token=token)
    api = HfApi(token=token)
    api.create_repo(
        REPO, repo_type="dataset", private=True, exist_ok=True
    )

    for filename in METADATA_FILES:
        api.upload_file(
            path_or_fileobj=str(SOURCE / filename),
            path_in_repo=filename,
            repo_id=REPO,
            repo_type="dataset",
        )

    with tarfile.open(ARCHIVE_PATH, "w") as archive:
        archive.add(SOURCE / "scenes", arcname="scenes")
        archive.add(SOURCE / "products", arcname="products")
    print(f"archive size: {ARCHIVE_PATH.stat().st_size / 1e9:.2f} GB")
    api.upload_file(
        path_or_fileobj=str(ARCHIVE_PATH),
        path_in_repo="images.tar",
        repo_id=REPO,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=README.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=REPO,
        repo_type="dataset",
    )
    print(f"uploaded private dataset: https://huggingface.co/datasets/{REPO}")


if __name__ == "__main__":
    main()
