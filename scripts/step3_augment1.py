"""
Step 3 — D4 对称增强 + Mild Random Crop（×8 扩充）
====================================================
功能:
  对 rgb_png/ 和 seg_id/ 做完全同步的 D4 Group Symmetry 增强
  + Mild Random Crop（大图 → 1024×1024）

  增强组合（×8）:
    0: 原图
    1: 旋转 90°
    2: 旋转 180°
    3: 旋转 270°
    4: 水平翻转
    5: 水平翻转 + 旋转 90°
    6: 水平翻转 + 旋转 180°
    7: 水平翻转 + 旋转 270°

  绝对禁止的操作（遥感语义保护）:
    ✗ Hue / 色相调整
    ✗ Gaussian Blur / 模糊
    ✗ Random Erasing / Cutout
    ✗ 大幅度缩放（会产生插值模糊）

输出结构:
  Golden_set/
    augmented/
      rgb/        ← 增强后的原图 (PNG)
      seg_id/     ← 增强后的 ID mask (PNG)

命名规则:
  原文件名_aug{0-7}.png
  例: tile_001_aug3.png

用法:
  python step3_augment.py --root /path/to/Golden_set
  python step3_augment.py --root /path/to/Golden_set --crop_size 1024 --no_crop
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ── 增强配置 ─────────────────────────────────────────────
CROP_SIZE   = 1024    # 裁剪目标尺寸
CROP_SCALE  = 1.0     # 裁剪时保持原尺寸（如原图已是 1024，设为 1.0）
RANDOM_SEED = 42

# D4 变换定义: (水平翻转?, 旋转次数×90°)
D4_TRANSFORMS = [
    (False, 0),   # 原图
    (False, 1),   # 旋转 90
    (False, 2),   # 旋转 180
    (False, 3),   # 旋转 270
    (True,  0),   # 水平翻转
    (True,  1),   # 水平翻转 + 旋转 90
    (True,  2),   # 水平翻转 + 旋转 180
    (True,  3),   # 水平翻转 + 旋转 270
]


def apply_d4(img: np.ndarray, flip: bool, rot_k: int,
             interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    """应用单个 D4 变换，mask 用 NEAREST 插值"""
    if flip:
        img = cv2.flip(img, 1)  # 水平翻转
    if rot_k > 0:
        img = np.rot90(img, k=rot_k)
    return img


def random_crop(rgb: np.ndarray, mask: np.ndarray, size: int):
    """
    Mild Random Crop：从大图随机裁出 size×size 的区域
    rgb 和 mask 用完全相同的随机起点，保证对齐
    """
    h, w = rgb.shape[:2]
    if h <= size and w <= size:
        # 图太小，直接 pad 到 size
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        rgb  = cv2.copyMakeBorder(rgb,  0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=255)
        return rgb, mask

    max_y = h - size
    max_x = w - size
    y = random.randint(0, max_y)
    x = random.randint(0, max_x)
    return rgb[y:y+size, x:x+size], mask[y:y+size, x:x+size]


def augment_pair(rgb_path: Path, mask_path: Path,
                 out_rgb_dir: Path, out_mask_dir: Path,
                 crop_size: int, do_crop: bool):
    """对一对 (rgb, mask) 生成 8 个增强版本"""
    rgb  = cv2.imread(str(rgb_path),  cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if rgb is None:
        print(f"  ⚠️  无法读取: {rgb_path}")
        return 0
    if mask is None:
        print(f"  ⚠️  无法读取: {mask_path}")
        return 0

    # 尺寸一致性检查
    if rgb.shape[:2] != mask.shape[:2]:
        print(f"  ⚠️  尺寸不一致: {rgb_path.name} rgb={rgb.shape[:2]} mask={mask.shape[:2]}")
        # 强制对齐 mask 到 rgb
        mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    stem = rgb_path.stem
    count = 0

    for aug_idx, (flip, rot_k) in enumerate(D4_TRANSFORMS):
        # 应用 D4 变换
        aug_rgb  = apply_d4(rgb,  flip, rot_k, cv2.INTER_LINEAR)
        aug_mask = apply_d4(mask, flip, rot_k, cv2.INTER_NEAREST)

        # Mild Random Crop（大图才做，小图不做）
        if do_crop:
            h, w = aug_rgb.shape[:2]
            if h > crop_size or w > crop_size:
                aug_rgb, aug_mask = random_crop(aug_rgb, aug_mask, crop_size)

        out_name = f"{stem}_aug{aug_idx}.png"
        cv2.imwrite(str(out_rgb_dir  / out_name), aug_rgb)
        cv2.imwrite(str(out_mask_dir / out_name), aug_mask)
        count += 1

    return count


def verify_augmentation(out_dir: Path, n_sample: int = 4):
    """生成增强效果可视化（展示同一张图的8种变换）"""
    import matplotlib.pyplot as plt

    rgb_files  = sorted((out_dir / "rgb").glob("*_aug0.png"))
    if not rgb_files:
        return

    # 取第一张的 8 个变换
    base_stem  = rgb_files[0].stem.replace("_aug0", "")
    rgb_dir    = out_dir / "rgb"
    mask_dir   = out_dir / "seg_id"

    fig, axes = plt.subplots(2, 8, figsize=(24, 6))
    fig.suptitle(f"D4 增强验证: {base_stem}", fontsize=13)

    aug_names = [
        "原图", "旋转90°", "旋转180°", "旋转270°",
        "水平翻转", "翻转+90°", "翻转+180°", "翻转+270°"
    ]

    for i in range(8):
        rgb_f  = rgb_dir  / f"{base_stem}_aug{i}.png"
        mask_f = mask_dir / f"{base_stem}_aug{i}.png"

        if rgb_f.exists():
            img = cv2.imread(str(rgb_f))
            axes[0][i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0][i].set_title(aug_names[i], fontsize=8)
        axes[0][i].axis("off")

        if mask_f.exists():
            mask = cv2.imread(str(mask_f), cv2.IMREAD_GRAYSCALE)
            axes[1][i].imshow(mask, cmap="tab10", vmin=0, vmax=9)
        axes[1][i].axis("off")

    axes[0][0].set_ylabel("RGB", fontsize=10)
    axes[1][0].set_ylabel("Mask ID", fontsize=10)

    plt.tight_layout()
    out_path = out_dir / "augmentation_verify.png"
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    print(f"🖼️  增强验证图: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",      required=True)
    parser.add_argument("--crop_size", type=int, default=CROP_SIZE)
    parser.add_argument("--no_crop",   action="store_true", help="禁用裁剪（原图已是目标尺寸时用）")
    parser.add_argument("--seed",      type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    root      = Path(args.root)
    rgb_dir   = root / "rgb_png"
    mask_dir  = root / "seg_id"
    out_dir   = root / "augmented"

    out_rgb_dir  = out_dir / "rgb"
    out_mask_dir = out_dir / "seg_id"
    out_rgb_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  Step 3 — D4 增强 ×8")
    print("=" * 55)
    print(f"  裁剪尺寸: {args.crop_size}×{args.crop_size}")
    print(f"  裁剪开关: {'关闭' if args.no_crop else '开启'}")

    # 收集配对文件
    rgb_files  = sorted(rgb_dir.glob("*.png"))
    pairs = []
    missing_mask = []
    for rf in rgb_files:
        mf = mask_dir / rf.name
        if mf.exists():
            pairs.append((rf, mf))
        else:
            missing_mask.append(rf.name)

    if missing_mask:
        print(f"\n  ⚠️  以下 rgb 文件没有对应 seg_id（已跳过）:")
        for n in missing_mask:
            print(f"     {n}")

    print(f"\n  找到配对: {len(pairs)} 张")
    print(f"  预计输出: {len(pairs) * 8} 张")

    total_out = 0
    for rgb_f, mask_f in tqdm(pairs, desc="增强中"):
        n = augment_pair(rgb_f, mask_f,
                         out_rgb_dir, out_mask_dir,
                         args.crop_size, not args.no_crop)
        total_out += n

    print(f"\n✅ 增强完成！共生成 {total_out} 张")
    print(f"   输出目录: {out_dir}")
    print(f"   rgb:    {out_rgb_dir}")
    print(f"   seg_id: {out_mask_dir}")

    # 验证可视化
    verify_augmentation(out_dir)

    print(f"\n下一步: python step4_train_loop1.py --root {root}")


if __name__ == "__main__":
    main()