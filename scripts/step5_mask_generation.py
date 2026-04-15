import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

from sam3.model.sam3_image_processor import Sam3Processor

# 复用 step4 的模型构建逻辑（包含 LoRA checkpoint 兼容）
from step4_evaluation import _build_models


# 与 step4 一致的类别定义与颜色
CATEGORY_COLORS: Dict[int, Tuple[int, int, int]] = {
	1: (0, 0, 255),      # road
	2: (0, 225, 255),    # water
	3: (0, 255, 0),      # foliage
	4: (255, 0, 0),      # building
	5: (128, 0, 128),    # grass
}

DEFAULT_LABEL_PROMPTS: Dict[int, str] = {
	1: "road",
	2: "water",
	3: "foliage",
	4: "building",
	5: "grass",
}


def load_label_thresholds(threshold_json_path: str) -> Dict[int, float]:
	with open(threshold_json_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	rec = data.get("recommended_threshold", {})
	if not rec:
		raise ValueError(f"未在阈值文件中找到 recommended_threshold: {threshold_json_path}")

	out: Dict[int, float] = {}
	for k, v in rec.items():
		cat_id = int(k)
		thr = float(v)
		if not (0.0 < thr < 1.0):
			raise ValueError(f"类别 {cat_id} 的阈值不在 (0,1): {thr}")
		out[cat_id] = thr
	return out


def infer_one_image(
	processor: Sam3Processor,
	image: Image.Image,
	label_prompts: Dict[int, str],
	label_thresholds: Dict[int, float],
) -> np.ndarray:
	"""对单张图做五类分割并返回 RGB 彩色 mask（H, W, 3）。"""

	state = processor.set_image(image, state={})
	h, w = image.height, image.width
	out_rgb = np.zeros((h, w, 3), dtype=np.uint8)

	# 固定顺序，保证可重复
	for cat_id in sorted(label_prompts.keys()):
		prompt = label_prompts[cat_id]
		thr = float(label_thresholds.get(cat_id, 0.5))

		state = processor.set_text_prompt(prompt, state)
		masks_prob = state.get("masks_logits", None)  # 变量名沿用 processor 实现，实际是 sigmoid 后概率
		if masks_prob is None or masks_prob.numel() == 0:
			continue

		# masks_prob: [N, 1, H, W]
		probs = masks_prob.squeeze(1)
		if probs.ndim != 3 or probs.shape[0] == 0:
			continue

		pred_mask = (probs > thr).any(dim=0).detach().cpu().numpy()
		color = CATEGORY_COLORS.get(cat_id)
		if color is None:
			continue
		out_rgb[pred_mask] = np.array(color, dtype=np.uint8)

	return out_rgb


def main():
	parser = argparse.ArgumentParser(
		description="Use exp5 best checkpoint + per-label thresholds to generate segmentation masks for DFC18 opt images."
	)
	parser.add_argument(
		"--config",
		default="/home/projectx/Sam3_data_engine/experiments/exp5/config.yaml",
		help="exp5 config yaml",
	)
	parser.add_argument(
		"--threshold-json",
		default="/home/projectx/Sam3_data_engine/experiments/exp5/visualization_label_threshold_custom/label_threshold_suggestion.json",
		help="Per-label threshold suggestion JSON",
	)
	parser.add_argument(
		"--finetune-ckpt",
		default="/home/projectx/Sam3_data_engine/experiments/exp5/checkpoints/best_train_loss.pt",
		help="Finetuned checkpoint path",
	)
	parser.add_argument(
		"--input-dir",
		default="/home/projectx/Sam3_data_engine/dataset/DFC18/DFC18/opt",
		help="Input RGB image folder",
	)
	parser.add_argument(
		"--output-dir",
		default="/home/projectx/Sam3_data_engine/dataset/DFC18/DFC18/seg",
		help="Output segmentation RGB mask folder",
	)
	parser.add_argument(
		"--device",
		default="cuda",
		help="cuda or cpu",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite existing output masks",
	)
	args = parser.parse_args()

	cfg = OmegaConf.load(args.config)

	finetune_ckpt = args.finetune_ckpt
	if not os.path.isfile(finetune_ckpt):
		raise FileNotFoundError(f"finetune checkpoint 不存在: {finetune_ckpt}")

	base_ckpt = cfg.paths.checkpoint_path
	if not os.path.isfile(base_ckpt):
		raise FileNotFoundError(f"base checkpoint 不存在: {base_ckpt}")

	label_thresholds = load_label_thresholds(args.threshold_json)
	label_prompts = dict(DEFAULT_LABEL_PROMPTS)

	device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
	print(f"[INFO] device = {device}")
	print("[INFO] label thresholds:")
	for cat_id in sorted(label_prompts.keys()):
		print(f"  label {cat_id} ({label_prompts[cat_id]}): {label_thresholds.get(cat_id, 0.5):.3f}")

	_, finetune_model = _build_models(
		cfg=cfg,
		device=device,
		base_ckpt=base_ckpt,
		finetune_ckpt=finetune_ckpt,
	)

	resolution = int(getattr(cfg.scratch, "resolution", 1008))
	processor = Sam3Processor(
		model=finetune_model,
		resolution=resolution,
		device=device.type,
		confidence_threshold=0.0,
	)

	input_dir = Path(args.input_dir)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	valid_ext = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".jp2"}
	image_paths = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_ext])
	if not image_paths:
		raise RuntimeError(f"输入目录无图像: {input_dir}")

	print(f"[INFO] total input images: {len(image_paths)}")
	generated = 0
	skipped = 0

	for img_path in tqdm(image_paths, desc="Generating masks"):
		out_name = img_path.stem + ".png"
		out_path = output_dir / out_name
		if out_path.exists() and not args.overwrite:
			skipped += 1
			continue

		image = Image.open(img_path).convert("RGB")
		mask_rgb = infer_one_image(
			processor=processor,
			image=image,
			label_prompts=label_prompts,
			label_thresholds=label_thresholds,
		)
		Image.fromarray(mask_rgb, mode="RGB").save(out_path)
		generated += 1

	print(f"[OK] generated={generated}, skipped={skipped}, output_dir={output_dir}")


if __name__ == "__main__":
	main()
