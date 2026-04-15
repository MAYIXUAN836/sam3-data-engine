import json
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pycocotools import mask as maskUtils


# augmented2/seg_id 当前是灰度图，不是 1..5 的类别 ID。
# 这些值来自原始彩色标签经灰度读取后的近似取值。
GRAY_VALUES_BY_CAT = {
    # 经 us3d_500/us3d-500-refined.json 与 seg_id 像素逐点对齐校验：
    # gray 29 -> road, gray 76 -> building（原版本这两类写反）
    1: [29],         # road
    2: [161],        # water
    3: [149, 150],   # foliage
    4: [76],         # building
    5: [52, 53],     # grass
}

def generate_coco_json(rgb_dir, mask_dir, out_json):
    rgb_dir = Path(rgb_dir)
    mask_dir = Path(mask_dir)
    
    categories = [
        {'id': 1, 'name': 'road', 'supercategory': ''},
        {'id': 2, 'name': 'water', 'supercategory': ''},
        {'id': 3, 'name': 'foliage', 'supercategory': ''},
        {'id': 4, 'name': 'building', 'supercategory': ''},
        {'id': 5, 'name': 'grass', 'supercategory': ''}
    ]
    
    images = []
    annotations = []
    ann_id = 1
    
    rgb_files = sorted(list(rgb_dir.glob('*.png')) + list(rgb_dir.glob('*.jpg')) + list(rgb_dir.glob('*.tif')))
    
    for img_id, img_path in enumerate(tqdm(rgb_files, desc="Generating JSON"), 1):
        # 找对应的mask
        mask_path = mask_dir / (img_path.stem + ".png")
        if not mask_path.exists():
            continue
            
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]
        
        images.append({
            "id": img_id,
            "width": w,
            "height": h,
            "file_name": img_path.name
        })
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
        
        # 为每个类别提取掩码（按灰度调色板映射）
        for cat in categories:
            cat_id = cat['id']
            valid_vals = GRAY_VALUES_BY_CAT.get(cat_id, [])
            if not valid_vals:
                continue
            binary_mask = np.isin(mask, valid_vals).astype(np.uint8)
            if binary_mask.sum() == 0:
                continue
            
            # 使用pycocotools编码RLE
            rle = maskUtils.encode(np.asfortranarray(binary_mask))
            rle['counts'] = rle['counts'].decode('utf-8')
            area = float(maskUtils.area(rle))
            bbox = maskUtils.toBbox(rle).tolist()
            
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "segmentation": rle,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
                "attributes": {}
            })
            ann_id += 1
            
    out_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    with open(out_json, 'w') as f:
        json.dump(out_dict, f)
        
    print(f"Saved {len(annotations)} annotations to {out_json}")

if __name__ == '__main__':
    generate_coco_json(
        rgb_dir="/home/projectx/Sam3_data_engine/dataset/us3d-500/augmented3/rgb",
        mask_dir="/home/projectx/Sam3_data_engine/dataset/us3d-500/augmented3/seg_id",
        out_json="/home/projectx/Sam3_data_engine/dataset/us3d-500/augmented3/segmentation_augmented3.json"
    )
