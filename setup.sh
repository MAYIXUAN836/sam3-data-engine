#!/usr/bin/env bash
# setup.sh for Sam3_data_engine
# 快速初始化环境和依赖，可选下载权重

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-sam3}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_VERSION="${TORCH_VERSION:-2.7.0}"
DOWNLOAD_WEIGHTS="${DOWNLOAD_WEIGHTS:-1}"
VERIFY_DATASETS="${VERIFY_DATASETS:-1}"
SETUP_DATASETS="${SETUP_DATASETS:-1}"
DOWNLOAD_DATASETS="${DOWNLOAD_DATASETS:-0}"
DATASET_HF_REPO="${DATASET_HF_REPO:-YixuanMa/sam3-data-engine-dataset}"
HF_REPO="${HF_REPO:-YixuanMa/sam3-data-engine-checkpoints}"
DOWNLOAD_BASE_WEIGHTS="${DOWNLOAD_BASE_WEIGHTS:-1}"
DOWNLOAD_FINETUNED_EXP5="${DOWNLOAD_FINETUNED_EXP5:-1}"
SAM3_CKPT_FILES=("sam3.pt" "model.safetensors")
EXP5_CKPT_FILES=(
    "experiments/exp5/checkpoints/best_train_loss.pt"
    "experiments/exp5/checkpoints/latest.pt"
    "experiments/exp5/config.yaml"
    "experiments/exp5/config_resolved.yaml"
)

echo "📦 Setting up Sam3_data_engine environment..."
echo "  Environment: ${ENV_NAME}"
echo "  Python: ${PYTHON_VERSION}"
echo "  PyTorch: ${TORCH_VERSION}"
echo "  Download Weights: ${DOWNLOAD_WEIGHTS}"
echo "  Download Datasets from HF: ${DOWNLOAD_DATASETS}"
echo "  Dataset HF Repo: ${DATASET_HF_REPO}"
echo "  Verify Datasets: ${VERIFY_DATASETS}"
echo "  Setup Datasets: ${SETUP_DATASETS}"
echo "  Download Base Weights: ${DOWNLOAD_BASE_WEIGHTS}"
echo "  Download Finetuned exp5: ${DOWNLOAD_FINETUNED_EXP5}"

# Load conda
source /home/projectx/miniconda/etc/profile.d/conda.sh || {
    echo "⚠️ conda not found. Please ensure conda is installed."
    exit 1
}

# Create or activate conda environment
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "🔧 Creating conda environment: ${ENV_NAME}"
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
else
    echo "✓ Environment ${ENV_NAME} already exists"
fi

conda activate "${ENV_NAME}"
echo "✓ Activated environment: ${ENV_NAME}"

# Upgrade pip
echo "📥 Upgrading pip..."
pip install -U pip

# Install PyTorch with CUDA support
echo "📥 Installing PyTorch ${TORCH_VERSION} with CUDA 12.6..."
pip install "torch==${TORCH_VERSION}" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install package in development mode with training dependencies
echo "📥 Installing sam3 package with training dependencies..."
pip install -e "${ROOT_DIR}/sam3[train]"

# Ensure asset directories exist for downloaded checkpoints.
mkdir -p "${ROOT_DIR}/checkpoints"

# Optional: Download model weights from Hugging Face
if [[ "${DOWNLOAD_WEIGHTS}" == "1" ]]; then
    echo "📥 Downloading model weights from ${HF_REPO}..."
    ROOT_DIR_ENV="${ROOT_DIR}" \
    HF_REPO_ENV="${HF_REPO}" \
    DOWNLOAD_BASE_WEIGHTS_ENV="${DOWNLOAD_BASE_WEIGHTS}" \
    DOWNLOAD_FINETUNED_EXP5_ENV="${DOWNLOAD_FINETUNED_EXP5}" \
    /home/projectx/miniconda/bin/python - <<'PYTHON_EOF'
from pathlib import Path
import os
from huggingface_hub import hf_hub_download

repo_id = os.environ["HF_REPO_ENV"]
root_dir = Path(os.environ["ROOT_DIR_ENV"])
download_base = os.environ.get("DOWNLOAD_BASE_WEIGHTS_ENV", "1") == "1"
download_exp5 = os.environ.get("DOWNLOAD_FINETUNED_EXP5_ENV", "1") == "1"

base_out_dir = root_dir / "checkpoints"
base_out_dir.mkdir(parents=True, exist_ok=True)

if download_base:
    for filename in ("sam3.pt", "model.safetensors"):
        print(f"[hf] downloading {filename} from {repo_id}")
        downloaded = hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename=filename,
            local_dir=str(base_out_dir),
            local_dir_use_symlinks=False,
        )
        print(f"[hf] ready: {downloaded}")

if download_exp5:
    exp5_files = (
        "experiments/exp5/checkpoints/best_train_loss.pt",
        "experiments/exp5/checkpoints/latest.pt",
        "experiments/exp5/config.yaml",
        "experiments/exp5/config_resolved.yaml",
    )
    for filename in exp5_files:
        target_dir = root_dir / Path(filename).parent
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"[hf] downloading {filename} from {repo_id}")
        downloaded = hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename=filename,
            local_dir=str(root_dir),
            local_dir_use_symlinks=False,
        )
        print(f"[hf] ready: {downloaded}")

if not download_base and not download_exp5:
    print("[hf] skipped all checkpoints by config")
PYTHON_EOF
else
    echo "⏭️ Skipping weight download (DOWNLOAD_WEIGHTS=0)"
fi

# Optional: Download dataset snapshot from Hugging Face dataset repo.
if [[ "${DOWNLOAD_DATASETS}" == "1" ]]; then
    echo "📥 Downloading dataset snapshot from ${DATASET_HF_REPO}..."
    ROOT_DIR_ENV="${ROOT_DIR}" \
    DATASET_HF_REPO_ENV="${DATASET_HF_REPO}" \
    /home/projectx/miniconda/bin/python - <<'PYTHON_EOF'
import os
from huggingface_hub import snapshot_download

root_dir = os.environ["ROOT_DIR_ENV"]
repo_id = os.environ["DATASET_HF_REPO_ENV"]

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=root_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)

print(f"[hf] dataset snapshot synced from {repo_id} to {root_dir}")
PYTHON_EOF
else
    echo "⏭️ Skipping dataset download from HF (DOWNLOAD_DATASETS=0)"
fi

if [[ "${VERIFY_DATASETS}" == "1" ]]; then
    echo "🔎 Verifying dataset layout..."
    missing=0
    for required in "${ROOT_DIR}/dataset/Golden_set" "${ROOT_DIR}/dataset/us3d-500" "${ROOT_DIR}/dataset/DFC18"; do
        if [[ -d "${required}" ]]; then
            echo "✓ ${required##*/} present"
        else
            echo "⚠️ ${required##*/} missing"
            missing=1
        fi
    done
    if [[ "${missing}" == "1" ]]; then
        echo "ℹ️ These datasets are not redistributed in this repo. Use the download_datasets.sh script for help."
    fi
fi

# Optional: Setup/verify datasets
if [[ "${SETUP_DATASETS}" == "1" ]] && [[ -f "${ROOT_DIR}/download_datasets.sh" ]]; then
    echo ""
    echo "📋 Running dataset verification & setup assistant..."
    bash "${ROOT_DIR}/download_datasets.sh"
fi

echo ""
echo "✅ Setup complete for ${ENV_NAME}!"
echo ""
echo "Next steps:"
echo "  1. Activate environment: conda activate ${ENV_NAME}"
echo "  2. Check installation: python -c 'import sam3; print(sam3.__version__)'"
echo "  3. Run examples: jupyter notebook sam3/examples/"
echo ""
echo "For training:"
echo "  python sam3/train.py -c sam3/train/configs/your_config.yaml"
echo ""
echo "Optional: Download weights after setup"
echo "  DOWNLOAD_WEIGHTS=1 bash setup.sh"
echo "Optional: Download dataset from Hugging Face dataset repo"
echo "  DOWNLOAD_DATASETS=1 DATASET_HF_REPO=YixuanMa/sam3-data-engine-dataset bash setup.sh"
echo "Optional: Download only finetuned exp5 checkpoints"
echo "  DOWNLOAD_BASE_WEIGHTS=0 DOWNLOAD_FINETUNED_EXP5=1 bash setup.sh"
echo "Optional: Download only base checkpoints"
echo "  DOWNLOAD_BASE_WEIGHTS=1 DOWNLOAD_FINETUNED_EXP5=0 bash setup.sh"
echo "Optional: Skip dataset setup"
echo "  SETUP_DATASETS=0 bash setup.sh"
echo "Optional: Disable dataset checks"
echo "  VERIFY_DATASETS=0 bash setup.sh"
echo ""
