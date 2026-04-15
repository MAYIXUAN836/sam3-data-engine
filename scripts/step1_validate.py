"""
Step 1 — 数据验证脚本
=====================================
功能:
  1. 检查 rgb_png / seg 文件一一对应
  2. 从 segmentation.json 自动提取颜色映射
  3. 检查 seg 里每张图用到了哪些颜色（类别）
  4. 输出每类像素占比分布（类别不平衡预警）
  5. 生成可视化报告 validation_report.png

用法:
  python step1_validate.py --root /path/to/Golden_set
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 类别名称（顺序与 COCO categories 对应）─────────────────
CLASS_NAMES = ["Building", "Road", "Water", "Foliage", "Grass"]


def load_color_map(json_path: Path):
    """
    从 segmentation.json 提取 category_id -> RGB 颜色映射。
    CVAT 导出的 COCO 格式通常在 categories[i]['color'] 里存颜色。
    如果没有 color 字段，则打印警告并返回 None。
    """
    with open(json_path) as f:
        data = json.load(f)

    categories = data.get("categories", [])
    color_map = {}  # category_id -> (R, G, B)

    for cat in categories:
        cid  = cat["id"]
        name = cat.get("name", f"class_{cid}")
        color_str = cat.get("color", None)

        if color_str and color_str.startswith("#"):
            r = int(color_str[1:3], 16)
            g = int(color_str[3:5], 16)
            b = int(color_str[5:7], 16)
            color_map[cid] = {"name": name, "rgb": (r, g, b)}
        else:
            print(f"  [WARN] category '{name}' (id={cid}) 没有 color 字段")

    return color_map, data


def collect_unique_colors(seg_dir: Path, sample_n: int = 10):
    """扫描前 sample_n 张 seg 图，收集所有出现过的 RGB 颜色"""
    all_colors = set()
    files = sorted(seg_dir.glob("*.png"))[:sample_n]
    for f in files:
        img = cv2.imread(str(f))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        pixels = img_rgb.reshape(-1, 3)
        unique = set(map(tuple, pixels))
        all_colors |= unique
    return all_colors


def compute_class_distribution(seg_dir: Path, color_map: dict):
    """
    统计每张 seg 图的类别像素占比，返回 per-class 平均占比 dict
    color_map: {cat_id: {"name":..., "rgb": (R,G,B)}}
    """
    rgb_to_name = {v["rgb"]: v["name"] for v in color_map.values()}
    class_pixel_counts = defaultdict(int)
    total_pixels = 0

    seg_files = sorted(seg_dir.glob("*.png"))
    per_image_dist = []

    for f in seg_files:
        img = cv2.imread(str(f))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        total = h * w
        total_pixels += total

        img_dist = {}
        for rgb, name in rgb_to_name.items():
            mask = np.all(img_rgb == np.array(rgb), axis=-1)
            cnt = int(mask.sum())
            class_pixel_counts[name] += cnt
            img_dist[name] = cnt / total
        per_image_dist.append(img_dist)

    avg_dist = {k: v / total_pixels for k, v in class_pixel_counts.items()}
    return avg_dist, per_image_dist


def check_pairing(rgb_dir: Path, seg_dir: Path):
    """检查 rgb_png 和 seg 的文件名一一对应"""
    rgb_stems = {f.stem for f in rgb_dir.glob("*.png")}
    seg_stems = {f.stem for f in seg_dir.glob("*.png")}

    only_rgb = rgb_stems - seg_stems
    only_seg = seg_stems - rgb_stems
    paired   = rgb_stems & seg_stems

    return sorted(paired), sorted(only_rgb), sorted(only_seg)


def plot_report(avg_dist, color_map, paired, only_rgb, only_seg, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Golden Set — 数据验证报告", fontsize=15, fontweight="bold")

    # ── 左图：类别像素占比 ─────────────────────────────────
    ax = axes[0]
    names  = list(avg_dist.keys())
    values = [avg_dist[n] * 100 for n in names]
    rgb_to_name = {v["rgb"]: k for k, v in color_map.items()}

    bar_colors = []
    for name in names:
        matched = [v["rgb"] for v in color_map.values() if v["name"] == name]
        if matched:
            r, g, b = matched[0]
            bar_colors.append((r/255, g/255, b/255))
        else:
            bar_colors.append("steelblue")

    bars = ax.barh(names, values, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("像素占比 (%)")
    ax.set_title("各类别平均像素占比（50张）")
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=10)
    ax.set_xlim(0, max(values) * 1.3)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # ── 右图：文件配对状态 ─────────────────────────────────
    ax2 = axes[1]
    ax2.axis("off")
    status_lines = [
        f"✅  配对成功：{len(paired)} 张",
        f"{'❌' if only_rgb else '✅'}  仅在 rgb_png（无对应 seg）：{len(only_rgb)} 张",
        f"{'❌' if only_seg else '✅'}  仅在 seg（无对应 rgb）：{len(only_seg)} 张",
        "",
        "── 颜色映射（来自 segmentation.json）──",
    ]
    for v in color_map.values():
        r, g, b = v["rgb"]
        status_lines.append(f"  ● {v['name']:12s}  #{r:02X}{g:02X}{b:02X}  ({r},{g},{b})")

    if only_rgb:
        status_lines += ["", "缺少 seg 的文件:"] + [f"  {s}" for s in only_rgb[:5]]
    if only_seg:
        status_lines += ["", "缺少 rgb 的文件:"] + [f"  {s}" for s in only_seg[:5]]

    ax2.text(0.05, 0.95, "\n".join(status_lines),
             transform=ax2.transAxes, fontsize=11,
             va="top", ha="left", family="monospace",
             bbox=dict(boxstyle="round", facecolor="#f5f5f5", alpha=0.8))
    ax2.set_title("文件配对 & 颜色映射")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\n📊 验证报告已保存至: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Golden_set 根目录")
    args = parser.parse_args()

    root     = Path(args.root)
    rgb_dir  = root / "rgb_png"
    seg_dir  = root / "seg"
    json_path= root / "segmentation.json"
    out_path = root / "validation_report.png"

    print("=" * 55)
    print("  Step 1 — 数据验证")
    print("=" * 55)

    # 1. 检查目录
    for d in [rgb_dir, seg_dir, json_path]:
        if not d.exists():
            print(f"❌ 找不到: {d}")
            return
        print(f"✅ 找到: {d}")

    # 2. 颜色映射
    print("\n── 从 segmentation.json 提取颜色映射 ──")
    color_map, _ = load_color_map(json_path)
    if not color_map:
        print("❌ segmentation.json 没有 color 字段，请手动在脚本顶部定义 COLOR_MAP")
        return
    for cid, v in color_map.items():
        print(f"  category {cid}: {v['name']:12s} → RGB{v['rgb']}")

    # 3. 文件配对
    print("\n── 文件配对检查 ──")
    paired, only_rgb, only_seg = check_pairing(rgb_dir, seg_dir)
    print(f"  配对成功: {len(paired)} 张")
    if only_rgb: print(f"  ❌ 仅有rgb无seg: {only_rgb}")
    if only_seg: print(f"  ❌ 仅有seg无rgb: {only_seg}")

    # 4. 像素分布
    print("\n── 类别像素分布统计 ──")
    avg_dist, _ = compute_class_distribution(seg_dir, color_map)
    for name, ratio in sorted(avg_dist.items(), key=lambda x: -x[1]):
        bar = "█" * int(ratio * 50)
        print(f"  {name:12s} {ratio*100:5.1f}%  {bar}")

    # 5. 不平衡预警
    vals = list(avg_dist.values())
    if max(vals) / (min(vals) + 1e-6) > 10:
        print("\n  ⚠️  最大/最小类别像素比 > 10x，建议在 Loss 里加类别权重")
    else:
        print("\n  ✅ 类别分布基本均衡")

    # 6. 生成报告图
    plot_report(avg_dist, color_map, paired, only_rgb, only_seg, out_path)
    print("\n✅ 验证完成，请查看 validation_report.png")


if __name__ == "__main__":
    main()