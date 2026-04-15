# Deployment Guide

本文档说明如何把 Sam3_data_engine 发布到 GitHub 和 Hugging Face。

## 前置条件

- GitHub 账号和已设置 SSH key 或 personal access token
- Hugging Face 账号，已装 `huggingface_hub` 包
- 本地已初始化并配置 Git

## Step 1: GitHub 仓库初始化

```bash
cd /home/projectx/Sam3_data_engine

# 初始化 Git 仓库
git init
git config user.name "Your Name"
git config user.email "your.email@example.com"

# 添加所有文件（.gitignore 已排除大文件和虚拟环境）
git add .

# 首次提交
git commit -m "Initial commit: SAM3 data engine for segmentation tasks"

# 改为主分支（GitHub 默认）
git branch -M main

# 添加远程仓库
# 仓库地址已写成 MAYIXUAN836
git remote add origin git@github.com:MAYIXUAN836/sam3-data-engine.git

# 推送到 GitHub
git push -u origin main
```

## Step 2: Hugging Face 模型仓库准备

### 创建 Hugging Face 模型仓库

```bash
pip install -U huggingface_hub

# 登录
huggingface-cli login

# 创建模型仓库（暂时可为空，后续训练生成权重后再上传）
huggingface-cli repo create sam3-data-engine-checkpoints --type model
```

### 上传权重文件（训练完成后）

当完成模型训练后，使用以下命令上传：

```bash
# 假设权重存放在 checkpoints/ 目录
huggingface-cli upload YixuanMa/sam3-data-engine-checkpoints \
  /path/to/checkpoint.pth \
  checkpoint.pth
```

## Step 3: 验证与后续配置

### 验证 GitHub 仓库

```bash
cd /home/projectx/Sam3_data_engine
git log --oneline
git remote -v
```

### 验证 Hugging Face 仓库

访问 `https://huggingface.co/YixuanMa/sam3-data-engine-checkpoints`

### 更新 setup.sh 中的 HF_REPO（如需自动下载权重）

编辑 `setup.sh`：

```bash
# 第 10 行附近，默认已经指向你的 HF 仓库
HF_REPO="${HF_REPO:-YixuanMa/sam3-data-engine-checkpoints}"
```

## Step 4: 用户快速开始（验证流程）

```bash
# 克隆代码仓库
git clone git@github.com:MAYIXUAN836/sam3-data-engine.git
cd sam3-data-engine

# 一键装环境
bash setup.sh

# 验证安装
conda activate sam3
python -c "import sam3; print(sam3.__version__)"

# 运行示例
jupyter notebook sam3/examples/sam3_image_predictor_example.ipynb
```

## 模型权重管理

### 首次上传权重

```bash
python << 'EOF'
from huggingface_hub import HfApi

api = HfApi()
repo_id = "YixuanMa/sam3-data-engine-checkpoints"

# 例如上传新训练的模型
  api.upload_file(
    path_or_fileobj="/path/to/your/checkpoint.pth",
    path_in_repo="model_v1.pth",
    repo_id=repo_id,
    repo_type="model"
  )

print("✅ Checkpoint uploaded successfully!")
EOF
```

### 更新 setup.sh 以支持自动下载

当权重可用后，编辑 `setup.sh` 中的权重下载部分：

```bash
if [[ "${DOWNLOAD_WEIGHTS}" == "1" ]]; then
    echo "📥 Downloading model weights from ${HF_REPO}..."
    mkdir -p "${ROOT_DIR}/checkpoints"
    
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download('${HF_REPO}', 'model_v1.pth', local_dir='${ROOT_DIR}/checkpoints')"
fi
```

## 常见问题

**Q: 能不能在 GitHub 里正常包含 pyproject.toml？**  
A: 可以，pyproject.toml 和其他配置文件都应该在 GitHub 里。.gitignore 会自动忽略 *.pth 等大权重文件。

**Q: 用户克隆后怎么验证环境是否正确安装？**  
A: 运行 `python -c "import sam3; print(sam3.__version__)"` 或 `pytest sam3/tests/` 如果有测试的话。

**Q: 权重文件太大上传到 HF 是否会出问题？**  
A: Hugging Face 支持单文件最大 50GB，支持断点续传。对于典型的深度学习权重（通常 < 2GB）完全可行。

**Q: 怎样禁用自动权重下载或跳过数据检查？**  
A: 用户运行 `DOWNLOAD_WEIGHTS=0 bash setup.sh` 即可跳过权重下载；运行 `VERIFY_DATASETS=0 bash setup.sh` 可跳过数据检查。

---

更新者：[Your Name]  
最后更新：2026-04-15
