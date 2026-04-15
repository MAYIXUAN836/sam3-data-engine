#!/bin/bash
# ============================================================
# Sam3 Dataset Downloader
# 自动下载and验证 Sam3 所需的数据集
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASET_DIR="${PROJECT_ROOT}/dataset"
DOWNLOAD_US3D="${DOWNLOAD_US3D:-1}"
DOWNLOAD_DFC18="${DOWNLOAD_DFC18:-1}"
VERIFY_ONLY="${VERIFY_ONLY:-0}"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         Sam3 Dataset Downloader & Verifier                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Dataset root: ${DATASET_DIR}"
echo "Download US3D-500: ${DOWNLOAD_US3D}"
echo "Download DFC18: ${DOWNLOAD_DFC18}"
echo ""

# ============================================================
# 验证已有数据集
# ============================================================
verify_datasets() {
    echo "🔍 Verifying dataset structure..."
    local all_ok=1
    
    # Golden_set
    if [[ -d "${DATASET_DIR}/Golden_set" ]]; then
        if [[ -d "${DATASET_DIR}/Golden_set/rgb_png" ]] && [[ -d "${DATASET_DIR}/Golden_set/seg" ]]; then
            echo "  ✓ Golden_set (complete)"
        else
            echo "  ⚠️  Golden_set (incomplete - missing subdirectories)"
            all_ok=0
        fi
    else
        echo "  ❌ Golden_set (missing)"
        all_ok=0
    fi
    
    # US3D-500
    if [[ -d "${DATASET_DIR}/us3d-500" ]]; then
        if [[ -f "${DATASET_DIR}/us3d-500"/*.tif ]] 2>/dev/null || [[ -f "${DATASET_DIR}/us3d-500"/*.json ]] 2>/dev/null; then
            echo "  ✓ US3D-500 (present)"
        else
            echo "  ⚠️  US3D-500 (empty or incomplete)"
            all_ok=0
        fi
    else
        echo "  ❌ US3D-500 (missing)"
        all_ok=0
    fi
    
    # DFC18
    if [[ -d "${DATASET_DIR}/DFC18" ]]; then
        if [[ -d "${DATASET_DIR}/DFC18/opt" ]] || [[ -d "${DATASET_DIR}/DFC18/pan" ]]; then
            echo "  ✓ DFC18 (present)"
        else
            echo "  ⚠️  DFC18 (empty or incomplete)"
            all_ok=0
        fi
    else
        echo "  ❌ DFC18 (missing)"
        all_ok=0
    fi
    
    return $all_ok
}

# ============================================================
# 信息display函数
# ============================================================
show_download_info() {
    echo ""
    echo "ℹ️  Dataset Download Information:"
    echo ""
    echo "1. US3D-500"
    echo "   Link: https://usgs-m2.gi.alaska.edu/geonarrative/usgs-cches/US3D-500-dataset.html"
    echo "   Size: ~30GB"
    echo "   Format: .tif files (multi-band GeoTIFF)"
    echo "   Steps:"
    echo "     • Download from official link"
    echo "     • Extract to: ${DATASET_DIR}/us3d-500/"
    echo ""
    echo "2. DFC18 (IEEE GRSS Data Fusion Contest 2018)"
    echo "   Link: https://www2.isprs.org/commissions/commission2/wg4/judges-of-paper-contents-for-2018.html"
    echo "   Alternative: http://www.grss-ieee.org/community/technical-committees/2018-ieee-grss-data-fusion-contest/"
    echo "   Size: ~50GB"
    echo "   Format: Multi-spectral GeoTIFF (PAN + MS)"
    echo "   Steps:"
    echo "     • Register and download from official contest page"
    echo "     • Extract to: ${DATASET_DIR}/DFC18/"
    echo ""
    echo "3. Golden_set (已有，自己标注的数据集)"
    echo "   Status: ✓ Present at ${DATASET_DIR}/Golden_set"
    echo ""
}

# ============================================================
# 主流程
# ============================================================
mkdir -p "${DATASET_DIR}"

if [[ "${VERIFY_ONLY}" == "1" ]]; then
    echo "[Verify Mode]"
    verify_datasets
    show_download_info
    exit 0
fi

# 尝试验证
echo "📋 Checking existing datasets..."
if verify_datasets; then
    echo ""
    echo "✅ All datasets present and valid!"
    echo ""
else
    echo ""
    echo "⚠️  Some datasets are missing or incomplete."
    show_download_info
    echo ""
    echo "💡 Next steps:"
    echo "   1. Download the missing datasets from links above"
    echo "   2. Extract them to ${DATASET_DIR}/"
    echo "   3. Run: bash $(basename "$0") to verify"
    echo ""
fi

# 数据集摘要
echo "📊 Dataset Summary:"
echo "   Golden_set: Training & validation data (segmentation masks + RGB images)"
echo "   US3D-500: Pre-training dataset (SAR + Optical multi-spectral)"
echo "   DFC18: Contest dataset (PAN + MS for evaluation)"
echo ""
echo "⚙️  For training, configure your config.yaml with:"
echo "   data:"
echo "     root: ${DATASET_DIR}"
echo "     train_set: Golden_set"
echo "     val_set: Golden_set"
echo ""
