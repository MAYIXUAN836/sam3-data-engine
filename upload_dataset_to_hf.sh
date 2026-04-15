#!/usr/bin/env bash
# Upload local Sam3 dataset directory to Hugging Face dataset repository.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="${ROOT_DIR}/dataset"
DATASET_HF_REPO="${DATASET_HF_REPO:-YixuanMa/sam3-data-engine-dataset}"
HF_TOKEN="${HF_TOKEN:-}"

if [[ ! -d "${DATASET_DIR}" ]]; then
    echo "❌ dataset directory not found: ${DATASET_DIR}"
    exit 1
fi

echo "📤 Uploading ${DATASET_DIR} to Hugging Face dataset repo: ${DATASET_HF_REPO}"
echo "   This can take a long time for large files."

if [[ -z "${HF_TOKEN}" ]]; then
    echo "ℹ️ HF_TOKEN not provided. Will use local huggingface-cli login session if available."
fi

ROOT_DIR_ENV="${ROOT_DIR}" \
DATASET_HF_REPO_ENV="${DATASET_HF_REPO}" \
HF_TOKEN_ENV="${HF_TOKEN}" \
/home/projectx/miniconda/bin/python - <<'PYTHON_EOF'
import os
from pathlib import Path
from huggingface_hub import HfApi

root_dir = Path(os.environ["ROOT_DIR_ENV"])
repo_id = os.environ["DATASET_HF_REPO_ENV"]
token = os.environ["HF_TOKEN_ENV"]

dataset_dir = root_dir / "dataset"

api = HfApi(token=token)
api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)

api.upload_large_folder(
    repo_id=repo_id,
    repo_type="dataset",
    folder_path=str(dataset_dir),
    path_in_repo="dataset",
)

print(f"[hf] upload completed: {repo_id}")
PYTHON_EOF

echo "✅ Upload finished: ${DATASET_HF_REPO}"
