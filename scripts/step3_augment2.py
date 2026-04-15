"""
Step 3 v2 — D4 + Scale + Brightness/Contrast + Grid Distortion + Mosaic
=========================================================================
轻量增强默认配置:
    1. D4 对称 ×8
    2. Restricted Random Scale: 0.8x / 1.0x / 1.2x  (×3)
    3. Brightness/Contrast 组合 ×3:
             (0.0,  0.0)  原始
             (+0.1, +0.1) 亮度+10%, 对比度+10%
             (-0.1, -0.1) 亮度-10%, 对比度-10%
    4. Grid Distortion (num_steps=3, distort_limit=0.05)
    5. Mosaic 4图拼接

总量预估 (500张, 随机采样模式):
    主增强目标(main_target_total)=7000张 (不再做全组合累乘)
    Grid Distortion  = 500张
    Mosaic           = 500张 (可调)
    合计             ≈ 8000张

绝对禁止:
  ✗ Hue / 色相调整
  ✗ Gaussian Blur
  ✗ Random Erasing / Cutout
  ✗ 大幅度缩放

输出:
  Golden_set/augmented2/rgb/
  Golden_set/augmented2/seg_id/

用法:
    python step3_augment_v2.py --root ~/Sam3_data_engine/Golden_set
    python step3_augment_v2.py --root ~/Sam3_data_engine/Golden_set --main_target_total 7000 --mosaic_n 100
    python step3_augment_v2.py --root ~/Sam3_data_engine/Golden_set --full_cartesian
"""


import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ── 配置 ──────────────────────────────────────────────────
CROP_SIZE   = 1024
RANDOM_SEED = 42

D4_TRANSFORMS = [
    (False, 0), (False, 1), (False, 2), (False, 3),
    (True,  0), (True,  1), (True,  2), (True,  3),
]

SCALE_FACTORS = [0.8, 1.0, 1.2]

# Brightness / Contrast 组合 (delta_brightness, delta_contrast)
BC_COMBOS = [
    ( 0.0,  0.0),   # 原始
    ( 0.1,  0.1),   # 亮度+10%, 对比度+10%
    (-0.1, -0.1),   # 亮度-10%, 对比度-10%
]

GRID_NUM_STEPS     = 3
GRID_DISTORT_LIMIT = 0.05

RGB_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MASK_SUFFIXES = [".png", ".tif", ".tiff"]


# ── 变换函数 ───────────────────────────────────────────────

def apply_d4(rgb, mask, flip, rot_k):
    if flip:
        rgb  = cv2.flip(rgb,  1)
        mask = cv2.flip(mask, 1)
    if rot_k > 0:
        rgb  = np.rot90(rgb,  k=rot_k)
        mask = np.rot90(mask, k=rot_k)
    return rgb, mask


def apply_scale_crop(rgb, mask, scale, crop_size):
    """缩放后随机裁剪到 crop_size×crop_size"""
    h, w = rgb.shape[:2]
    new_h = max(int(h * scale), crop_size)
    new_w = max(int(w * scale), crop_size)

    rgb_s  = cv2.resize(rgb,  (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_s = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    y = random.randint(0, new_h - crop_size)
    x = random.randint(0, new_w - crop_size)
    return rgb_s[y:y+crop_size, x:x+crop_size], mask_s[y:y+crop_size, x:x+crop_size]


def apply_brightness_contrast(rgb, delta_b, delta_c):
    """
    Brightness ±10%: 像素值乘以 (1 + delta_b)
    Contrast   ±10%: (像素值 - 均值) × (1 + delta_c) + 均值
    只作用于 RGB 图，mask 不变
    """
    if delta_b == 0.0 and delta_c == 0.0:
        return rgb.copy()

    img = rgb.astype(np.float32)

    if delta_b != 0.0:
        img = img * (1.0 + delta_b)

    if delta_c != 0.0:
        mean = img.mean()
        img  = (img - mean) * (1.0 + delta_c) + mean

    return np.clip(img, 0, 255).astype(np.uint8)


def apply_grid_distortion(rgb, mask, num_steps=3, distort_limit=0.05):
    """轻微几何扭曲，模拟不同传感器的几何畸变"""
    h, w = rgb.shape[:2]
    stepx = w // num_steps
    stepy = h // num_steps

    src_pts, dst_pts = [], []
    for i in range(num_steps + 1):
        for j in range(num_steps + 1):
            sx = min(j * stepx, w - 1)
            sy = min(i * stepy, h - 1)
            dx = int(distort_limit * stepx * (random.random() * 2 - 1))
            dy = int(distort_limit * stepy * (random.random() * 2 - 1))
            src_pts.append([sx, sy])
            dst_pts.append([np.clip(sx+dx, 0, w-1), np.clip(sy+dy, 0, h-1)])

    src_pts = np.float32(src_pts)
    dst_pts = np.float32(dst_pts)

    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    if M is not None:
        rgb_out  = cv2.warpPerspective(rgb,  M, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT)
        mask_out = cv2.warpPerspective(mask, M, (w, h),
                                       flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=255)
        return rgb_out, mask_out
    return rgb, mask


def make_mosaic(pairs, out_rgb_dir, out_mask_dir, crop_size, num_mosaic):
    """Mosaic 4图拼接"""
    n    = len(pairs)
    half = crop_size // 2
    count = 0

    for i in tqdm(range(num_mosaic), desc="Mosaic"):
        indices = random.choices(range(n), k=4)
        imgs, masks = [], []

        for idx in indices:
            rf, mf = pairs[idx]
            rgb  = cv2.imread(str(rf),  cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mf),  cv2.IMREAD_GRAYSCALE)
            if rgb is None:
                rgb  = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
                mask = np.full((crop_size, crop_size), 255, dtype=np.uint8)

            # 随机 D4
            rgb, mask = apply_d4(rgb, mask,
                                  random.choice([True, False]),
                                  random.randint(0, 3))
            # 随机 BC
            db, dc = random.choice(BC_COMBOS)
            rgb = apply_brightness_contrast(rgb, db, dc)

            rgb  = cv2.resize(rgb,  (half, half), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (half, half), interpolation=cv2.INTER_NEAREST)
            imgs.append(rgb)
            masks.append(mask)

        mosaic_rgb = np.concatenate([
            np.concatenate([imgs[0],  imgs[1]],  axis=1),
            np.concatenate([imgs[2],  imgs[3]],  axis=1),
        ], axis=0)
        mosaic_mask = np.concatenate([
            np.concatenate([masks[0], masks[1]], axis=1),
            np.concatenate([masks[2], masks[3]], axis=1),
        ], axis=0)

        name = f"mosaic_{i:04d}.png"
        cv2.imwrite(str(out_rgb_dir  / name), mosaic_rgb)
        cv2.imwrite(str(out_mask_dir / name), mosaic_mask)
        count += 1

    return count


def verify_samples(out_dir, n=4):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] 跳过验证图生成: matplotlib 不可用 ({e})")
        return
    rgb_files = sorted((out_dir / "rgb").glob("*.png"))
    if not rgb_files:
        return
    samples = random.sample(rgb_files, min(n, len(rgb_files)))
    fig, axes = plt.subplots(n, 2, figsize=(10, 5*n))
    if n == 1:
        axes = [axes]
    for i, f in enumerate(samples):
        rgb  = cv2.imread(str(f))
        mf   = out_dir / "seg_id" / f.name
        mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE) if mf.exists() else None
        axes[i][0].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        axes[i][0].set_title(f.name[:40], fontsize=7)
        axes[i][0].axis("off")
        if mask is not None:
            axes[i][1].imshow(mask, cmap="tab10", vmin=0, vmax=9)
        axes[i][1].axis("off")
    plt.tight_layout()
    out_path = out_dir / "augment_verify.png"
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    print(f"🖼️  验证图: {out_path}")


def collect_pairs(rgb_dir: Path, mask_dir: Path):
    """Collect RGB/mask pairs by stem, supporting tif/png/jpeg inputs."""
    pairs = []
    rgb_files = sorted(
        f for f in rgb_dir.iterdir()
        if f.is_file() and f.suffix.lower() in RGB_SUFFIXES
    )

    for rf in rgb_files:
        stem = rf.stem
        mf = None
        for ext in MASK_SUFFIXES:
            cand = mask_dir / f"{stem}{ext}"
            if cand.exists():
                mf = cand
                break
        if mf is not None:
            pairs.append((rf, mf))
        else:
            print(f"  ⚠️  无 mask: {rf.name}")

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",      required=True)
    parser.add_argument("--crop_size", type=int, default=CROP_SIZE)
    parser.add_argument("--main_target_total", type=int, default=7000)
    parser.add_argument("--grid_n",    type=int, default=500)
    parser.add_argument("--mosaic_n",  type=int, default=500)
    parser.add_argument("--full_cartesian", action="store_true")
    parser.add_argument("--seed",      type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    root     = Path(args.root)
    rgb_dir  = root / "rgb_png"
    mask_dir = root / "seg_id"
    out_dir  = root / "augmented2"

    out_rgb_dir  = out_dir / "rgb"
    out_mask_dir = out_dir / "seg_id"
    out_rgb_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Step 3 v2 — D4 + Scale + BC + GridDistortion + Mosaic")
    print("=" * 60)

    pairs = collect_pairs(rgb_dir, mask_dir)

    n  = len(pairs)
    full_main = n * len(D4_TRANSFORMS) * len(SCALE_FACTORS) * len(BC_COMBOS)
    n_main = full_main if args.full_cartesian else args.main_target_total
    n_main = max(0, n_main)
    min_main_to_cover_all_d4 = n * len(D4_TRANSFORMS)
    if not args.full_cartesian and n_main < min_main_to_cover_all_d4:
        print(
            f"  [INFO] main_target_total={n_main} 小于 D4 全覆盖下限 {min_main_to_cover_all_d4}，自动提升到该下限"
        )
        n_main = min_main_to_cover_all_d4
    print(f"\n  原始配对:      {n} 张")
    if args.full_cartesian:
        print(f"  主增强(全组合): {n}×{len(D4_TRANSFORMS)}×{len(SCALE_FACTORS)}×{len(BC_COMBOS)} = {n_main} 张")
    else:
        print(f"  主增强(随机采样): 目标 {n_main} 张 (含每图 D4×8 全覆盖, 全组合上限 {full_main} 张)")
    print(f"  Grid Distort:  {args.grid_n} 张")
    print(f"  Mosaic:        {args.mosaic_n} 张")
    print(f"  预计总量:      {n_main + args.grid_n + args.mosaic_n} 张\n")

    total = 0

    # ── 1. D4 + Scale + BC ───────────────────────────────
    print("[1/3] D4 + Scale + Brightness/Contrast ...")
    if args.full_cartesian:
        for rf, mf in tqdm(pairs, desc="D4+Scale+BC(full)"):
            rgb  = cv2.imread(str(rf),  cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mf),  cv2.IMREAD_GRAYSCALE)
            if rgb is None or mask is None:
                continue
            stem  = rf.stem
            aug_i = 0
            for flip, rot_k in D4_TRANSFORMS:
                d4_rgb, d4_mask = apply_d4(rgb, mask, flip, rot_k)
                for si, scale in enumerate(SCALE_FACTORS):
                    s_rgb, s_mask = apply_scale_crop(d4_rgb, d4_mask, scale, args.crop_size)
                    for bi, (db, dc) in enumerate(BC_COMBOS):
                        bc_rgb = apply_brightness_contrast(s_rgb, db, dc)
                        name = f"{stem}_d{aug_i}_s{si}_b{bi}.png"
                        cv2.imwrite(str(out_rgb_dir  / name), bc_rgb)
                        cv2.imwrite(str(out_mask_dir / name), s_mask)
                        total += 1
                aug_i += 1
    else:
        if n == 0:
            print("  [WARN] 没有可用配对，跳过主增强")
        else:
            # Phase A: 每张图固定生成 D4 的 8 个方向，确保 D4 全覆盖；scale/BC 随机采样。
            for rf, mf in tqdm(pairs, desc="D4(8)+random_tail(core)"):
                rgb = cv2.imread(str(rf), cv2.IMREAD_COLOR)
                mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
                if rgb is None or mask is None:
                    continue

                stem = rf.stem
                for d_i, (flip, rot_k) in enumerate(D4_TRANSFORMS):
                    si = random.randrange(len(SCALE_FACTORS))
                    bi = random.randrange(len(BC_COMBOS))
                    scale = SCALE_FACTORS[si]
                    db, dc = BC_COMBOS[bi]
                    d4_rgb, d4_mask = apply_d4(rgb, mask, flip, rot_k)
                    s_rgb, s_mask = apply_scale_crop(d4_rgb, d4_mask, scale, args.crop_size)
                    bc_rgb = apply_brightness_contrast(s_rgb, db, dc)
                    name = f"{stem}_core_d{d_i:02d}_s{si}_b{bi}.png"
                    cv2.imwrite(str(out_rgb_dir / name), bc_rgb)
                    cv2.imwrite(str(out_mask_dir / name), s_mask)
                    total += 1

            # Phase B: 追加随机样本，补足到 main_target_total。
            remain = max(0, n_main - total)
            for j in tqdm(range(remain), desc="D4(8)+random_tail(extra)"):
                rf, mf = random.choice(pairs)
                rgb = cv2.imread(str(rf), cv2.IMREAD_COLOR)
                mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
                if rgb is None or mask is None:
                    continue

                stem = rf.stem
                d_i = random.randrange(len(D4_TRANSFORMS))
                flip, rot_k = D4_TRANSFORMS[d_i]
                si = random.randrange(len(SCALE_FACTORS))
                bi = random.randrange(len(BC_COMBOS))
                scale = SCALE_FACTORS[si]
                db, dc = BC_COMBOS[bi]

                d4_rgb, d4_mask = apply_d4(rgb, mask, flip, rot_k)
                s_rgb, s_mask = apply_scale_crop(d4_rgb, d4_mask, scale, args.crop_size)
                bc_rgb = apply_brightness_contrast(s_rgb, db, dc)
                name = f"{stem}_extra_{j:05d}_d{d_i:02d}_s{si}_b{bi}.png"
                cv2.imwrite(str(out_rgb_dir / name), bc_rgb)
                cv2.imwrite(str(out_mask_dir / name), s_mask)
                total += 1
    print(f"  生成: {total} 张")

    # ── 2. Grid Distortion ────────────────────────────────
    print("\n[2/3] Grid Distortion ...")
    gd_count = 0
    for i in tqdm(range(args.grid_n), desc="GridDistort"):
        if n == 0:
            break
        rf, mf = random.choice(pairs)
        rgb  = cv2.imread(str(rf),  cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mf),  cv2.IMREAD_GRAYSCALE)
        if rgb is None or mask is None:
            continue
        aug_rgb, aug_mask = apply_d4(rgb, mask,
                                      random.choice([True, False]),
                                      random.randint(0, 3))
        gd_rgb, gd_mask = apply_grid_distortion(
            aug_rgb, aug_mask, GRID_NUM_STEPS, GRID_DISTORT_LIMIT)
        if gd_rgb.shape[0] != args.crop_size or gd_rgb.shape[1] != args.crop_size:
            gd_rgb, gd_mask = apply_scale_crop(gd_rgb, gd_mask, 1.0, args.crop_size)
        name = f"{rf.stem}_gd_{i:04d}.png"
        cv2.imwrite(str(out_rgb_dir  / name), gd_rgb)
        cv2.imwrite(str(out_mask_dir / name), gd_mask)
        gd_count += 1
    total += gd_count
    print(f"  生成: {gd_count} 张")

    # ── 3. Mosaic ─────────────────────────────────────────
    print(f"\n[3/3] Mosaic {args.mosaic_n} 张 ...")
    mc = make_mosaic(pairs, out_rgb_dir, out_mask_dir, args.crop_size, args.mosaic_n)
    total += mc
    print(f"  生成: {mc} 张")

    print(f"\n{'='*60}")
    print(f"✅ 完成！总计 {total} 张")
    print(f"   D4+Scale+BC:     {n_main} 张")
    print(f"   Grid Distortion: {gd_count} 张")
    print(f"   Mosaic:          {mc} 张")
    print(f"   输出: {out_dir}")

    verify_samples(out_dir, n=4)


if __name__ == "__main__":
    main()