#!/usr/bin/env bash
# setup.sh for Sam3_data_engine
# 快速初始化环境和依赖，可选下载权重

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-sam3}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_VERSION="${TORCH_VERSION:-2.7.0}"
DOWNLOAD_WEIGHTS="${DOWNLOAD_WEIGHTS:-0}"
HF_REPO="${HF_REPO:-YOUR_NAME/sam3-data-engine-checkpoints}"

echo "📦 Setting up Sam3_data_engine environment..."
echo "  Environment: ${ENV_NAME}"
echo "  Python: ${PYTHON_VERSION}"
echo "  PyTorch: ${TORCH_VERSION}"
echo "  Download Weights: ${DOWNLOAD_WEIGHTS}"

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

# Optional: Download model weights from Hugging Face
if [[ "${DOWNLOAD_WEIGHTS}" == "1" ]]; then
    echo "📥 Downloading model weights from ${HF_REPO}..."
    mkdir -p "${ROOT_DIR}/checkpoints"
    
    if ! python -c "import huggingface_hub" 2>/dev/null; then
        echo "⚠️ huggingface_hub not available, skipping weight download"
    else
        # Replace with actual checkpoint filenames when available
        echo "ℹ️ No checkpoints defined yet. Update setup.sh when checkpoints are available."
    fi
else
    echo "⏭️ Skipping weight download (DOWNLOAD_WEIGHTS=0)"
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
echo ""
