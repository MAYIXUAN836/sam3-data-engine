# Sam3 数据集设置指南 | Sam3 Dataset Guide

**更新时间**: 2026-04-15  
**状态**: ✅ 更新完成 - 现已支持数据集验证和辅助设置

---

## 📢 新增功能

Sam3_data_engine 现已升级，提供**数据集验证和智能辅助**功能：

- ✅ 自动验证数据集完整性
- ✅ 提供下载链接和说明
- ✅ 集成到 setup.sh 工作流

---

## 🚀 快速使用

### 方式 1: 通过 setup.sh 自动运行

```bash
git clone git@github.com:MAYIXUAN836/sam3-data-engine.git
cd sam3-data-engine
bash setup.sh
# ↑ 会自动验证数据集并显示下载指南
```

### 方式 2: 单独运行数据集脚本

```bash
# 验证数据集（自动检查并显示下载指南）
bash download_datasets.sh

# 仅显示验证信息和下载链接（不需要网络）
VERIFY_ONLY=1 bash download_datasets.sh

# 跳过数据集设置
SETUP_DATASETS=0 bash setup.sh
```

---

## 📊 Sam3 数据集清单

| 数据集 | 大小 | 用途 | 源 | 状态 |
|--------|------|------|-----|------|
| **Golden_set** | ~10GB | 训练和验证 | 自有标注 | ✓ 已有 |
| **US3D-500** | ~30GB | 预训练 | 公开数据集 | 需下载 |
| **DFC18** | ~50GB | 评估和测试 | IEEE 竞赛 | 需下载 |

---

## 📍 数据集详情

### 1. Golden_set (自有 - 已位置)

**位置**: `dataset/Golden_set/`  
**结构**:
```
Golden_set/
├── rgb_png/           ← 原始 RGB 图像 (.png)
├── seg/               ← 彩色掩码 (.png, CVAT 标注)
├── rgb_tif/           ← 原始多波段 GeoTIFF (.tif)
└── ...
```

**说明**: 已有的自己标注的数据集，包含 RGB 图像和分割掩码。用于 SAM3 的微调训练。

---

### 2. US3D-500 (需下载)

**官方链接**: https://usgs-m2.gi.alaska.edu/geonarrative/usgs-cches/US3D-500-dataset.html  
**大小**: ~30GB  
**格式**: Multi-band GeoTIFF (.tif)  
**用途**: 预训练数据集

**下载步骤**:
```bash
# 1. 访问官方网站
# 2. 下载所有 .tif 文件
# 3. 提取到本地
mkdir -p dataset/us3d-500
# 4. 复制或移动所有.tif文件到该目录

# 5. 验证
bash download_datasets.sh
```

**预期目录结构**:
```
dataset/us3d-500/
├── *.tif              ← 多波段 GeoTIFF 文件
└── (其他相关文件)
```

---

### 3. DFC18 (需下载)

**官方链接**: https://www2.isprs.org/commissions/commission2/wg4/judges-of-paper-contents-for-2018.html  
**替代链接**: http://www.grss-ieee.org/community/technical-committees/2018-ieee-grss-data-fusion-contest/  
**大小**: ~50GB  
**格式**: 全色 (PAN) + 多光谱 (MS) GeoTIFF  
**用途**: 数据融合竞赛数据，用于评估

**下载步骤**:
```bash
# 1. 前往官方竞赛页面
# 2. 注册并同意许可
# 3. 下载所有数据文件
# 4. 提取到本地
mkdir -p dataset/DFC18
# 5. 将 PAN 和 MS 文件放入相应目录

# 6. 验证
bash download_datasets.sh
```

**预期目录结构**:
```
dataset/DFC18/
├── opt/               ← Optical 数据 (.tif)
├── pan/               ← Panchromatic .tif)
├── seg/               ← 参考分割掩码 (可选)
└── (其他相关文件)
```

---

## 🔧 配置选项

### setup.sh 中的数据集相关选项

```bash
# 完整安装（包括数据集验证）
bash setup.sh

# 跳过数据集设置流程
SETUP_DATASETS=0 bash setup.sh

# 跳过数据集验证（但仍进行其他设置）
VERIFY_DATASETS=0 bash setup.sh

# 同时跳过验证和设置
SETUP_DATASETS=0 VERIFY_DATASETS=0 bash setup.sh
```

### download_datasets.sh 中的选项

```bash
# 验证现有数据集（默认）
bash download_datasets.sh

# 仅显示验证信息，不运行主逻辑
VERIFY_ONLY=1 bash download_datasets.sh

# 放在后台运行时跳过某些数据集的下载
DOWNLOAD_US3D=0 bash download_datasets.sh     # 跳过 US3D-500
DOWNLOAD_DFC18=0 bash download_datasets.sh    # 跳过 DFC18
```

---

## 🎯 典型工作流

### 场景 1: 完整新服务器部署

```bash
# ① 克隆仓库
git clone git@github.com:MAYIXUAN836/sam3-data-engine.git
cd sam3-data-engine

# ② 创建环境和检查
bash setup.sh
# 输出会显示缺失的数据集和下载链接

# ③ 根据提示下载数据集（从官方源）
# ... 下载 US3D-500 和 DFC18 ...

# ④ 将数据集放到正确位置
mkdir -p dataset/{us3d-500,DFC18}
# cp /path/to/downloaded/US3D-500/*.tif dataset/us3d-500/
# cp /path/to/downloaded/DFC18/*.tif dataset/DFC18/

# ⑤ 验证数据集完整性
bash download_datasets.sh

# 输出应显示:
# ✓ Golden_set (complete)
# ✓ US3D-500 (present)
# ✓ DFC18 (present)
```

### 场景 2: 快速环境设置（跳过数据集)

```bash
# 只安装环境和权重，暂不处理数据集
bash setup.sh &
# 或
SETUP_DATASETS=0 VERIFY_DATASETS=0 bash setup.sh

# 之后手动处理数据集
# (稍后下载数据集并放置)
```

### 场景 3: 数据集已存在，仅验证

```bash
# 验证已有数据集的完整性
bash download_datasets.sh

# 或仅显示信息
VERIFY_ONLY=1 bash download_datasets.sh
```

---

## 📝 常见问题

### Q1: 如何知道数据集是否正确放置？

运行数据集验证脚本：
```bash
bash download_datasets.sh
```

输出示例：
```
✓ Golden_set (complete)    ← 完整
⚠️  US3D-500 (empty)       ← 空或不完整
❌ DFC18 (missing)         ← 缺失
```

### Q2: 激活 VERIFY_ONLY 后会发生什么？

```bash
VERIFY_ONLY=1 bash download_datasets.sh
```

这会显示完整的数据集信息和下载链接，**无需进行任何验证操作**。适合在网络受限或只想查看信息时使用。

### Q3: US3D-500 和 DFC18 太大，能否分部分下载？

可以。这两个数据集都是模块化的：

- **US3D-500**: 可以选择性下载特定地区的 .tif 文件
- **DFC18**: 可以只下载 PAN 或 MS 数据，无需全部

下载后直接放入对应目录即可。

### Q4: 如何跳过数据集检查？

```bash
# 方式 1: 跳过 setup.sh 中的数据集检查
SETUP_DATASETS=0 VERIFY_DATASETS=0 bash setup.sh

# 方式 2: 直接激活环境
conda activate sam3
python sam3/train.py -c config.yaml
```

### Q5: 能否使用符号链接挂载数据集？

可以。只要目录结构正确即可：

```bash
# 例如，从其他位置挂载 US3D-500
ln -s /mnt/nas/US3D-500 dataset/us3d-500

# 验证
bash download_datasets.sh
# 应显示: ✓ US3D-500 (present)
```

---

## 🔗 相关链接

| 资源 | 链接 |
|------|------|
| 官方 GitHub | https://github.com/MAYIXUAN836/sam3-data-engine |
| HuggingFace 权重 | https://huggingface.co/YixuanMa/sam3-data-engine-checkpoints |
| US3D-500 官网 | https://usgs-m2.gi.alaska.edu/geonarrative/usgs-cches/US3D-500-dataset.html |
| DFC18 竞赛页 | https://www2.isprs.org/commissions/commission2/wg4/ |

---

## 📞 需要帮助？

1. **验证数据集**: `bash download_datasets.sh`
2. **查看下载指南**: `VERIFY_ONLY=1 bash download_datasets.sh`
3. **查看训练脚本说明**: `cat scripts/readme.md`
4. **检查部署指南**: 查看主仓库中的 `DEPLOYMENT_GUIDE_CN.md`

---

**更新记录**:
- 2026-04-15: 添加数据集验证脚本和辅助工具
- 2026-04-15: 集成到 setup.sh 工作流
