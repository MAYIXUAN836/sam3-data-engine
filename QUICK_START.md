# Quick Start

## 三条命令快速开始

### 1. 克隆仓库
```bash
git clone git@github.com:MAYIXUAN836/sam3-data-engine.git
cd sam3-data-engine
```

### 2. 一键装环境（含 PyTorch、基础权重+exp5微调权重下载与数据校验）
```bash
bash setup.sh
```

### 3. 验证安装
```bash
conda activate sam3
python -c "import sam3; print(sam3.__version__)"
```

## 运行示例

### 图像分割

```bash
# 激活环境
conda activate sam3

# 启动 Jupyter
jupyter notebook sam3/examples/sam3_image_predictor_example.ipynb
```

### 视频分割

```bash
conda activate sam3
jupyter notebook sam3/examples/sam3_video_predictor_example.ipynb
```

### 批量推理

```bash
conda activate sam3
jupyter notebook sam3/examples/sam3_image_batched_inference.ipynb
```

## 环境要求

- **Python**: 3.12+
- **PyTorch**: 2.7.0+
- **CUDA**: 12.6+（仅 GPU 训练需要）

## 包含内容

- **SAM3 模型代码**: `sam3/`
  - 图像检测器 (DETR-based)
  - 视频追踪器 (Transformer-based)
  - 文本+可视化提示支持

- **数据引擎**: `sam3/data_engine/`
  - 270K+ 独特概念自动标注
  - 高质量分割数据集生成

- **训练脚本**: `sam3/train/`
  - Hydra 配置管理
  - 多卡/多节点训练支持

## 环境变量配置

### setup.sh 支持的环境变量：

```bash
# 自定义环境名
ENV_NAME=my_sam3 bash setup.sh

# 自定义 Python 版本
PYTHON_VERSION=3.11 bash setup.sh

# 自定义 PyTorch 版本
TORCH_VERSION=2.6.0 bash setup.sh

# 立刻下载权重（默认开启）
DOWNLOAD_WEIGHTS=1 bash setup.sh

# 自定义 HF 仓库（默认 YixuanMa/sam3-data-engine-checkpoints）
HF_REPO=YixuanMa/sam3-data-engine-checkpoints bash setup.sh

# 只下载 exp5 微调权重
DOWNLOAD_BASE_WEIGHTS=0 DOWNLOAD_FINETUNED_EXP5=1 bash setup.sh

# 只下载基础权重
DOWNLOAD_BASE_WEIGHTS=1 DOWNLOAD_FINETUNED_EXP5=0 bash setup.sh

# 关闭数据校验
VERIFY_DATASETS=0 bash setup.sh
```

## 训练

```bash
conda activate sam3

# 查看可用配置
ls -la sam3/train/configs/

# 单 GPU 训练
python sam3/train/train.py -c sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 0 --num-gpus 1

# 多 GPU 训练
python sam3/train/train.py -c sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 0 --num-gpus 4
```

## 常见问题

**Q: 装 PyTorch 时出现 CUDA 版本不匹配？**  
A: 使用 `INSTALL_CUDA_TORCH=0 bash setup.sh` 装 CPU 版本，或手动指定 CUDA 版本

**Q: 权重下载失败？**  
A: 手动从 HF 下载，或确保 `huggingface_hub` 已装且已认证

**Q: 训练内存不足？**  
A: 减小 batch_size 或使用梯度累积（参考配置文件）

---

详细说明请参考 [DEPLOY.md](DEPLOY.md)  
官方代码：[facebook/sam3](https://github.com/facebookresearch/sam3)
