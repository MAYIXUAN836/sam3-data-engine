"""
Step 2 (修复版) — 直接从 segmentation.json 生成单通道 ID mask
修复内容: 按类别优先级光栅化，Foliage 最后画，覆盖 Building 多标的边缘

类别 ID (0-based):
    0 = Road
    1 = Water
    2 = Foliage
    3 = Building
    4 = Grass
    255 = 忽略/背景

光栅化顺序 (数字越大越后画，会覆盖前面):
    Road     → 1
    Water    → 2
    Grass    → 3
    Building → 4
    Foliage  → 5  ← 最后画，覆盖 Building 多标边缘

用法:
    python step2_rgb_to_id.py --root ~/Sam3_data_engine/Golden_set
"""

import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from pycocotools import mask as coco_mask

# ── 类别映射 (0-based ID) ──────────────────────────────────
CATEGORY_MAP = {
    "road":     0,
    "water":    1,
    "foliage":  2,
    "building": 3,
    "grass":    4,
}

# ── 光栅化优先级（数字越大越后画，会覆盖前面）─────────────
PAINT_ORDER = {
    "road":     1,
    "water":    2,
    "grass":    3,
    "building": 4,
    "foliage":  5,  # 最后画，覆盖 Building 多标的边缘
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    root    = Path(args.root)
    out_dir = root / "seg_id"
    out_dir.mkdir(exist_ok=True)

    with open(root / "segmentation_fixed.json") as f:
        data = json.load(f)

    # category_id -> class_id (0-based)
    cat_id_to_class = {}
    # category_id -> name (用于排序)
    cat_id_to_name = {}
    for cat in data["categories"]:
        name = cat["name"].lower()
        cat_id_to_name[cat["id"]] = name
        if name in CATEGORY_MAP:
            cat_id_to_class[cat["id"]] = CATEGORY_MAP[name]

    # image_id -> annotations
    img_to_anns = defaultdict(list)
    for ann in data["annotations"]:
        img_to_anns[ann["image_id"]].append(ann)

    for img_info in tqdm(data["images"], desc="生成 ID mask"):
        img_id     = img_info["id"]
        h, w       = img_info["height"], img_info["width"]
        stem       = Path(img_info["file_name"]).stem
        stem_clean = stem.replace("_RGB_", "_")
        out_path   = out_dir / f"{stem_clean}.png"

        id_mask = np.full((h, w), 255, dtype=np.uint8)  # 255 = 忽略

        # ── 按优先级排序，Foliage 最后画 ──────────────────
        anns_sorted = sorted(
            img_to_anns[img_id],
            key=lambda a: PAINT_ORDER.get(
                cat_id_to_name.get(a["category_id"], ""), 0)
        )

        for ann in anns_sorted:
            class_id = cat_id_to_class.get(ann["category_id"])
            if class_id is None:
                continue

            seg = ann["segmentation"]

            # RLE 格式
            if isinstance(seg, dict):
                rle = seg
                if isinstance(rle["counts"], list):
                    rle = coco_mask.frPyObjects(rle, h, w)
                binary = coco_mask.decode(rle).astype(bool)

            # Polygon 格式
            elif isinstance(seg, list):
                binary = np.zeros((h, w), dtype=bool)
                for poly in seg:
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                    tmp = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(tmp, [pts], 1)
                    binary |= tmp.astype(bool)
            else:
                continue

            id_mask[binary] = class_id

        cv2.imwrite(str(out_path), id_mask)

    print(f"\n✅ 完成！生成 {len(data['images'])} 张 ID mask → {out_dir}")
    print(f"\n光栅化顺序（后画的覆盖前面）:")
    for name, order in sorted(PAINT_ORDER.items(), key=lambda x: x[1]):
        print(f"  {order}. {name:12s} → ID={CATEGORY_MAP.get(name, '?')}")
    print(f"\n下一步: python step3_augment.py --root {root}")


if __name__ == "__main__":
    main()