# Project X — 数据处理 & 训练脚本

## 脚本执行顺序

```
step1_validate.py      ← 验证数据完整性
    ↓
step2_rgb_to_id.py     ← RGB mask → 单通道 ID 图
    ↓
step3_augment.py       ← D4 增强 ×8（50张 → 400张）
    ↓
step4_train_loop1.py   ← SAM3 Loop 1 微调
```

---

## 快速开始

```bash
# 1. 安装依赖
pip install torch torchvision opencv-python numpy matplotlib tqdm

# 2. 数据验证（先跑这个，确认数据没问题）
python step1_validate.py --root /path/to/Golden_set

# 3. RGB mask → ID 图
python step2_rgb_to_id.py --root /path/to/Golden_set

# 4. D4 增强
python step3_augment.py --root /path/to/Golden_set

# 5. 训练 Loop 1
python step4_train_loop1.py \
  --data_root /path/to/Golden_set \
  --sam_checkpoint /path/to/sam2_hiera_large.pt \
  --output_dir checkpoints/loop1 \
  --batch_size 8 \
  --epochs 50
```

---

## 目录结构（期望）

```
Golden_set/
├── rgb/                  ← 原始 .tif 多波段（不直接用）
├── rgb_png/              ← 训练原图 PNG ✅
├── seg/                  ← RGB 彩色 mask（CVAT 导出）✅
├── seg_id/               ← 单通道 ID mask（step2 生成）
├── depth/                ← US3D GT depth（Flux 阶段用）
├── depth_png/            ← depth PNG 版
├── canny/                ← 自生成边缘图（Flux 阶段用）
├── visualization/        ← CVAT 可视化叠加图
├── segmentation.json     ← COCO 格式标注 ✅
├── color_map.json        ← 颜色映射（step2 生成）
├── augmented/            ← 增强数据（step3 生成）
│   ├── rgb/
│   └── seg_id/
├── val/                  ← 永久验证集（手动放10张，不参与训练！）
│   ├── rgb/
│   └── seg_id/
├── validation_report.png ← step1 生成的验证报告
└── checkpoints/
    └── loop1/
        ├── best.pth
        ├── last.pth
        └── training_log.json
```

---

## 注意事项

### step2: 颜色匹配失败怎么办

如果 `validation_report.png` 里看到大量"未匹配像素"，
在 `step2_rgb_to_id.py` 顶部手动填写颜色映射，然后用 `--manual` 参数运行：

```python
MANUAL_COLOR_MAP = {
    0: ("Building", (128,   0,   0)),   # 从 CVAT 里看每个类别的颜色
    1: ("Road",     (128, 128, 128)),
    2: ("Water",    (  0,   0, 128)),
    3: ("Foliage",  (  0, 128,   0)),
    4: ("Grass",    (128, 128,   0)),
}
```

```bash
python step2_rgb_to_id.py --root /path/to/Golden_set --manual --tolerance 10
```

### step3: 原图已经是 1024×1024

```bash
python step3_augment.py --root /path/to/Golden_set --no_crop
```

### step4: SAM2 加载方式

在 `step4_train_loop1.py` 的 `setup_model()` 里取消注释对应的加载方式：

**官方 SAM2:**
```python
from sam2.build_sam import build_sam2
self.sam = build_sam2("sam2_hiera_l.yaml", self.args.sam_checkpoint, device=self.device)
```

**自定义 SAM3:**
```python
from your_sam3_module import SAM3
self.sam = SAM3()
self.sam.load_state_dict(torch.load(self.args.sam_checkpoint))
```

---

## 类别 ID 对照表

| ID | 类别     | 说明         |
|----|----------|-------------|
| 0  | Building | 建筑物       |
| 1  | Road     | 道路         |
| 2  | Water    | 水体         |
| 3  | Foliage  | 树木/植被    |
| 4  | Grass    | 草地         |
| 255| Ignore   | 忽略/背景    |

---

## Loop 2 / Loop 3

完成 Loop 1 训练后，用生成的 `best.pth` 对新图批量推理：

```bash
# 批量推理（用 Loop 1 模型标注新的 300 张图）
# → 导入 CVAT 人工修正
# → 合并数据集
# → 运行 Loop 2 训练（增加 LoRA r=32、Neck、Hard Negative Loss）
```

Loop 2 和 Loop 3 的训练脚本逻辑相同，只需调整：
- `--lora_r 32`
- 在 `setup_model()` 里解冻 Neck 层
- 在 `CombinedLoss` 里加类别权重

有问题直接在 Claude 里问！