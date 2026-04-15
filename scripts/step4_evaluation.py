import argparse
import csv
import itertools
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf

from sam3.model.utils.misc import copy_data_to_device
from sam3.train.utils.checkpoint_utils import load_state_dict_into_model


# Golden_set 五类语义对应的固定颜色 (R, G, B)
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


def _build_prompt_to_category(label_prompts: Dict[int, str]) -> Dict[str, int]:
	"""构建 prompt 文本到类别 id 的映射（小写、去首尾空白）。"""

	mapping: Dict[str, int] = {}
	for cat_id, prompt in label_prompts.items():
		key = str(prompt).lower().strip()
		if key:
			mapping[key] = int(cat_id)
	return mapping


def _parse_label_prompt_overrides(
	overrides: Optional[List[str]],
	base_prompts: Dict[int, str],
) -> Dict[int, str]:
	"""解析命令行 label prompt 覆盖项。

	格式: --label-prompt 1:road 或 --label-prompt 2=water
	"""

	resolved = dict(base_prompts)
	if not overrides:
		return resolved

	for item in overrides:
		text = str(item).strip()
		sep = ":" if ":" in text else "=" if "=" in text else None
		if sep is None:
			raise ValueError(
				f"label-prompt 格式错误: {item}，请使用 '<label_id>:<prompt>' 或 '<label_id>=<prompt>'"
			)

		cat_str, prompt = text.split(sep, 1)
		cat_str = cat_str.strip()
		prompt = prompt.strip()
		if not cat_str.isdigit():
			raise ValueError(f"label-prompt 类别 id 非整数: {item}")
		cat_id = int(cat_str)
		if cat_id <= 0:
			raise ValueError(f"label-prompt 类别 id 必须 > 0: {item}")
		if not prompt:
			raise ValueError(f"label-prompt prompt 不能为空: {item}")

		resolved[cat_id] = prompt

	return resolved


def _parse_label_threshold_overrides(
	overrides: Optional[List[str]],
	default_threshold: float,
) -> Dict[int, float]:
	"""解析命令行 label threshold 覆盖项。

	格式: --label-threshold 1:0.6 或 --label-threshold 2=0.75
	"""

	resolved: Dict[int, float] = {}
	if not overrides:
		return resolved

	for item in overrides:
		text = str(item).strip()
		sep = ":" if ":" in text else "=" if "=" in text else None
		if sep is None:
			raise ValueError(
				f"label-threshold 格式错误: {item}，请使用 '<label_id>:<threshold>' 或 '<label_id>=<threshold>'"
			)

		cat_str, thr_str = text.split(sep, 1)
		cat_str = cat_str.strip()
		thr_str = thr_str.strip()
		if not cat_str.isdigit():
			raise ValueError(f"label-threshold 类别 id 非整数: {item}")
		cat_id = int(cat_str)
		if cat_id <= 0:
			raise ValueError(f"label-threshold 类别 id 必须 > 0: {item}")

		try:
			thr = float(thr_str)
		except ValueError as e:
			raise ValueError(f"label-threshold 阈值非数字: {item}") from e

		if not (0.0 < thr < 1.0):
			raise ValueError(
				f"label-threshold 阈值必须在 (0,1) 区间内: {item}"
			)

		resolved[cat_id] = thr

	return resolved


def _parse_threshold_candidates(candidates_text: str) -> List[float]:
	"""解析阈值候选列表字符串，例如 '0.5,0.6,0.7'。"""

	items = [x.strip() for x in str(candidates_text).split(",") if x.strip()]
	if not items:
		raise ValueError("threshold-candidates 不能为空")

	vals: List[float] = []
	for item in items:
		try:
			v = float(item)
		except ValueError as e:
			raise ValueError(f"threshold-candidates 中存在非数字: {item}") from e
		if not (0.0 < v < 1.0):
			raise ValueError(f"threshold-candidates 值需在 (0,1): {v}")
		vals.append(v)

	# 去重并排序，避免重复计算
	return sorted(set(vals))


def _compute_iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> Optional[float]:
	"""计算两个 bool mask 的 IoU。若并集为空返回 None。"""

	device = mask_a.device if mask_a.is_cuda else (mask_b.device if mask_b.is_cuda else mask_a.device)
	a = mask_a.to(device=device, dtype=torch.bool)
	b = mask_b.to(device=device, dtype=torch.bool)
	union = (a | b).sum().item()
	if union == 0:
		return None
	inter = (a & b).sum().item()
	return float(inter / union)


def _rank_threshold_combinations(
	label_mean_iou: Dict[int, Dict[float, float]],
	combo_top_n_per_label: int,
	combo_top_k: int,
) -> List[Dict[str, object]]:
	"""基于每个 label 的 mean IoU，构造并排序阈值组合。

	说明：这里的组合分数为各 label mean IoU 的平均值（可分解目标）。
	为避免组合爆炸，仅取每个 label 前 N 个候选参与笛卡尔积。
	"""

	if combo_top_n_per_label <= 0:
		raise ValueError(f"combo_top_n_per_label 必须 > 0，当前为: {combo_top_n_per_label}")
	if combo_top_k <= 0:
		raise ValueError(f"combo_top_k 必须 > 0，当前为: {combo_top_k}")

	cat_ids = sorted(label_mean_iou.keys())
	if not cat_ids:
		return []

	# 每个类别先按 mean_iou 降序排序，仅保留前 N。
	ranked_by_cat: Dict[int, List[Tuple[float, float]]] = {}
	for cat_id in cat_ids:
		items = [(thr, iou) for thr, iou in label_mean_iou[cat_id].items()]
		if not items:
			return []
		items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
		ranked_by_cat[cat_id] = items_sorted[:combo_top_n_per_label]

	combos: List[Dict[str, object]] = []
	iterables = [ranked_by_cat[cat_id] for cat_id in cat_ids]
	for choice in itertools.product(*iterables):
		# choice: [(thr_cat1, iou_cat1), (thr_cat2, iou_cat2), ...]
		mean_iou = sum(x[1] for x in choice) / len(choice)
		thresholds = {str(cat_id): float(choice[i][0]) for i, cat_id in enumerate(cat_ids)}
		label_iou = {str(cat_id): float(choice[i][1]) for i, cat_id in enumerate(cat_ids)}
		combos.append(
			{
				"thresholds": thresholds,
				"estimated_mean_iou": float(mean_iou),
				"label_mean_iou": label_iou,
			}
		)

	combos.sort(key=lambda x: float(x["estimated_mean_iou"]), reverse=True)
	return combos[:combo_top_k]


def _auto_select_finetune_ckpt(ckpt_dir: str) -> Optional[str]:
	"""从 checkpoint 目录中自动选择一个 finetune checkpoint.

	优先选择最大的 `checkpoint_XX.pt`，否则退回到 `checkpoint.pt`。
	"""

	if not os.path.isdir(ckpt_dir):
		return None

	pattern = re.compile(r"checkpoint_(\d+)\.pt$")
	best_epoch = -1
	best_path: Optional[str] = None

	for name in os.listdir(ckpt_dir):
		m = pattern.match(name)
		if m is not None:
			epoch = int(m.group(1))
			if epoch > best_epoch:
				best_epoch = epoch
				best_path = os.path.join(ckpt_dir, name)

	if best_path is not None:
		return best_path

	ckpt_path = os.path.join(ckpt_dir, "checkpoint.pt")
	return ckpt_path if os.path.isfile(ckpt_path) else None


def _build_models(
	cfg,
	device: torch.device,
	base_ckpt: Optional[str],
	finetune_ckpt: str,
):
	"""按照训练 config 加载原始模型和 finetune 模型.

	- base_model: 使用 config 中的模型定义与 `base_ckpt` 权重；
	- finetune_model: 同一结构；若启用 LoRA 则先注入，再加载 finetune checkpoint。
	"""

	model_conf = cfg.trainer.model

	# 原始 SAM3 模型
	base_kwargs = {
		"eval_mode": True,
		"device": device.type,
	}
	if base_ckpt is not None:
		base_kwargs["checkpoint_path"] = base_ckpt

	base_model = instantiate(model_conf, **base_kwargs)
	base_model.to(device)
	base_model.eval()

	# finetune 模型：先构建裸模型 + LoRA，再加载训练 checkpoint
	finetune_model = instantiate(
		model_conf,
		checkpoint_path=None,
		load_from_HF=False,
		eval_mode=True,
		device=device.type,
	)

	# 先读 checkpoint，再根据权重结构自动判断是否需要 LoRA
	ckpt = torch.load(finetune_ckpt, map_location="cpu")
	state = ckpt.get("model", ckpt)
	has_lora_in_ckpt = any(
		k.endswith("lora_A") or k.endswith("lora_B") for k in state.keys()
	)

	# 优先使用 checkpoint 结构判定，避免 config 过期导致 key mismatch
	cfg_enable_lora = bool(getattr(cfg.trainer, "enable_lora", False))
	enable_lora = has_lora_in_ckpt
	if cfg_enable_lora != enable_lora:
		print(
			"  LoRA 自动判断: "
			f"config enable_lora={cfg_enable_lora}, "
			f"checkpoint has_lora={has_lora_in_ckpt}; 以 checkpoint 为准"
		)

	lora_r = int(getattr(cfg.trainer, "lora_r", 16))
	lora_alpha = float(getattr(cfg.trainer, "lora_alpha", 16.0))
	if enable_lora:
		# 从 checkpoint 的 lora_A 形状推断 rank，避免 r 配置不一致
		sample_lora_a_key = next(
			(k for k in state.keys() if k.endswith("lora_A")),
			None,
		)
		if sample_lora_a_key is not None:
			infer_r = int(state[sample_lora_a_key].shape[0])
			if infer_r > 0 and infer_r != lora_r:
				print(f"  LoRA rank 自动校正: cfg r={lora_r} -> ckpt r={infer_r}")
				lora_r = infer_r

		import sys as _sys

		script_dir = os.path.dirname(os.path.abspath(__file__))
		if script_dir not in _sys.path:
			_sys.path.insert(0, script_dir)
		from lora_injection import apply_lora_to_sam3

		apply_lora_to_sam3(finetune_model, r=lora_r, alpha=lora_alpha)
		print(f"  LoRA 注入: 已启用（r={lora_r}, alpha={lora_alpha}）")
	else:
		print("  LoRA 注入: 已禁用（checkpoint 不包含 LoRA 权重）")

	# 加载 finetune checkpoint
	load_state_dict_into_model(state_dict=state, model=finetune_model, strict=False)

	finetune_model.to(device)
	finetune_model.eval()

	return base_model, finetune_model


def _compute_pred_union_prob(stage_out: dict) -> Optional[torch.Tensor]:
	"""从模型输出构造 union 概率图（单张图，范围 [0,1]）。"""

	if "pred_masks" not in stage_out:
		return None
	pred_masks = stage_out["pred_masks"]

	if pred_masks.ndim == 4:
		if pred_masks.size(0) == 0 or pred_masks.size(1) == 0:
			return None
		prob = pred_masks.sigmoid()
		# 在 mask 维与 prompt 维取最大值，得到 union 概率
		return prob.amax(dim=1).amax(dim=0)
	if pred_masks.ndim == 3:
		if pred_masks.size(0) == 0:
			return None
		return pred_masks.sigmoid().amax(dim=0)
	return None


def _compute_test_loss(stage_out: dict, stage_target) -> float:
	"""计算单样本 test loss（Union Mask 的 BCE + Dice）。"""

	gt_union = _compute_gt_union_mask(stage_target)
	pred_union = _compute_pred_union_prob(stage_out)
	if gt_union is None or pred_union is None:
		return float("nan")

	gt_union = gt_union.to(device=pred_union.device, dtype=torch.float32)
	if pred_union.shape != gt_union.shape:
		pred_union = F.interpolate(
			pred_union.unsqueeze(0).unsqueeze(0),
			size=gt_union.shape,
			mode="bilinear",
			align_corners=False,
		)[0, 0]

	pred_union = pred_union.clamp(1e-6, 1.0 - 1e-6)
	bce = F.binary_cross_entropy(pred_union, gt_union)

	eps = 1e-6
	inter = (pred_union * gt_union).sum()
	dice = 1.0 - (2.0 * inter + eps) / (pred_union.sum() + gt_union.sum() + eps)

	return float((bce + dice).detach().cpu().item())


def _compute_gt_union_mask(target) -> Optional[torch.Tensor]:
	"""从 BatchedFindTarget 中构造 GT union mask（单张图）。"""

	segments = target.segments
	is_valid = target.is_valid_segment
	if segments is None or is_valid is None:
		return None
	if segments.numel() == 0:
		return None

	valid = is_valid.to(dtype=torch.bool)
	if valid.numel() == 0 or not bool(valid.any()):
		return None

	segs_valid = segments[valid]
	if segs_valid.ndim == 3:
		union = segs_valid.any(dim=0)
	elif segs_valid.ndim == 2:
		union = segs_valid
	else:
		return None

	return union.to(dtype=torch.float32)


def _compute_category_masks(target, metadata) -> Dict[int, torch.Tensor]:
	"""根据 BatchedFindTarget 和 BatchedInferenceMetadata 按类别构造 GT mask。

	返回一个字典: {category_id: mask}，mask 为 2D bool tensor。
	"""

	segments = target.segments
	is_valid = target.is_valid_segment
	num_boxes = target.num_boxes
	cat_ids = metadata.original_category_id

	if (
		segments is None
		or is_valid is None
		or num_boxes is None
		or cat_ids is None
	):
		return {}
	if segments.numel() == 0 or num_boxes.numel() == 0:
		return {}

	segments = segments.to(dtype=torch.bool)
	is_valid = is_valid.to(dtype=torch.bool)
	cat_ids = cat_ids.to(dtype=torch.long)

	cat_masks: Dict[int, torch.Tensor] = {}
	offset = 0
	for qi in range(num_boxes.shape[0]):
		n = int(num_boxes[qi].item())
		if n <= 0:
			continue

		cat_id = int(cat_ids[qi].item())
		if offset + n > segments.shape[0]:
			break

		segs_slice = segments[offset : offset + n]
		valid_slice = is_valid[offset : offset + n]
		offset += n

		if cat_id <= 0 or not bool(valid_slice.any()):
			continue

		union = segs_slice[valid_slice].any(dim=0)
		if cat_id in cat_masks:
			cat_masks[cat_id] |= union
		else:
			cat_masks[cat_id] = union

	return cat_masks


def _compute_pred_union_mask(
	stage_out: dict,
	pred_threshold: float = 0.5,
) -> Optional[torch.Tensor]:
	"""从模型输出中构造预测 union mask（单张图），合并所有 text prompt。"""

	if "pred_masks" not in stage_out:
		return None
	pred_masks = stage_out["pred_masks"]

	# 常见形状为 [N_prompt, N_mask, H, W]，也兼容 [N_mask, H, W]
	if pred_masks.ndim == 4:
		if pred_masks.size(0) == 0 or pred_masks.size(1) == 0:
			return None
		prob = pred_masks.sigmoid()
		binary = prob > pred_threshold
		# 先在 mask 维度上做 OR，再在 prompt 维度上做 OR
		union = binary.any(dim=1).any(dim=0).to(dtype=torch.float32)
		return union
	elif pred_masks.ndim == 3:
		if pred_masks.size(0) == 0:
			return None
		prob = pred_masks.sigmoid()
		binary = prob > pred_threshold
		union = binary.any(dim=0).to(dtype=torch.float32)
		return union
	else:
		return None


def _compute_pred_category_masks(
	stage_out: dict,
	batched_dp,
	prompt_to_category: Dict[str, int],
	label_thresholds: Dict[int, float],
	pred_threshold: float = 0.5,
) -> Dict[int, torch.Tensor]:
	"""根据 pred_masks 和文本 prompt，按类别构造预测 mask（不依赖 GT）。

	返回 {category_id: mask}，其中 mask 为 2D bool tensor（模型输出分辨率）。
	"""

	if "pred_masks" not in stage_out:
		return {}

	pred_masks = stage_out["pred_masks"]
	if pred_masks.ndim != 4:
		return {}

	B, Q, H, W = pred_masks.shape
	if B == 0 or Q == 0:
		return {}

	# text_ids 映射到 find_text_batch 里的具体 prompt 文本
	find_input = batched_dp.find_inputs[0]
	text_ids = find_input.text_ids
	text_batch = batched_dp.find_text_batch

	cat_masks: Dict[int, torch.Tensor] = {}
	for b in range(min(B, text_ids.shape[0])):
		text_idx = int(text_ids[b].item())
		if text_idx < 0 or text_idx >= len(text_batch):
			continue
		prompt_str = text_batch[text_idx].lower().strip()
		cat_id = prompt_to_category.get(prompt_str)
		if cat_id is None:
			continue
		thr = float(label_thresholds.get(cat_id, pred_threshold))

		masks_b = pred_masks[b]  # [Q, H, W]
		if masks_b.numel() == 0:
			continue
		prob = masks_b.sigmoid()
		binary = prob > thr
		union_b = binary.any(dim=0)  # [H, W] bool

		if cat_id in cat_masks:
			cat_masks[cat_id] |= union_b
		else:
			cat_masks[cat_id] = union_b

	return cat_masks


def _resize_to_image(mask: Optional[torch.Tensor], h: int, w: int) -> Optional[torch.Tensor]:
	"""将 2D mask resize 到原图大小 (h, w)。"""

	if mask is None:
		return None
	if mask.ndim != 2:
		return None
	mask_4d = mask.unsqueeze(0).unsqueeze(0).float()
	with torch.no_grad():
		out = F.interpolate(mask_4d, size=(h, w), mode="nearest")
	return out[0, 0]


def _resize_category_masks(
	cat_masks: Dict[int, torch.Tensor], h: int, w: int
) -> Dict[int, torch.Tensor]:
	"""将按类别的 mask 字典 resize 到原图大小。"""

	resized: Dict[int, torch.Tensor] = {}
	for cat_id, mask in cat_masks.items():
		mask_resized = _resize_to_image(mask.float(), h, w)
		if mask_resized is not None:
			resized[cat_id] = mask_resized > 0.5
	return resized


def _plot_triplet(
	image,
	gt_cat_masks: Dict[int, torch.Tensor],
	base_cat_masks: Dict[int, torch.Tensor],
	finetune_cat_masks: Dict[int, torch.Tensor],
	out_path: str,
):
	"""使用 CATEGORY_COLORS 绘制 GT / 原始 / finetune 三个 overlay 图并保存。

	三列分别使用各自的 mask 字典，不再与 GT 相交：
	- GT: ground truth;
	- Original: 原始 SAM3 预测;
	- Finetuned: finetuned SAM3 预测。
	"""

	import numpy as np

	os.makedirs(os.path.dirname(out_path), exist_ok=True)

	img_np = np.array(image).astype("float32")

	def make_panel(cat_masks: Dict[int, torch.Tensor]) -> np.ndarray:
		panel = img_np.copy()
		for cat_id, mask_t in cat_masks.items():
			color = CATEGORY_COLORS.get(int(cat_id))
			if color is None:
				continue

			mask = mask_t.cpu()
			if mask.ndim != 2:
				continue
			mask_np = mask.cpu().numpy().astype(bool)
			if not mask_np.any():
				continue

			color_arr = np.array(color, dtype="float32")
			alpha = 0.5
			panel[mask_np] = panel[mask_np] * (1.0 - alpha) + color_arr * alpha
		return panel.astype("uint8")

	gt_panel = make_panel(gt_cat_masks) if gt_cat_masks else img_np.astype("uint8")
	base_panel = make_panel(base_cat_masks) if base_cat_masks else img_np.astype("uint8")
	finetune_panel = (
		make_panel(finetune_cat_masks)
		if finetune_cat_masks
		else img_np.astype("uint8")
	)

	fig, axes = plt.subplots(1, 3, figsize=(15, 5))
	for ax, panel, title in zip(
		axes,
		[gt_panel, base_panel, finetune_panel],
		["GT", "Original SAM3", "Fine-tuned SAM3"],
	):
		ax.imshow(panel)
		ax.set_title(title)
		ax.axis("off")

	plt.tight_layout()
	fig.savefig(out_path, dpi=150)
	plt.close(fig)


def _build_isolated_val_annotation(
	source_ann_file: str,
	val_img_folder: str,
	out_ann_file: str,
) -> Tuple[str, int, int]:
	"""将原始 val 标注映射到 val/rgb，可用样本写入新标注文件。

	处理规则：
	- 优先保留原 file_name；
	- 若找不到，则尝试将 `*_RGB_*.tif` 映射为 `*_*_*.png`；
	- 仅保留在 `val_img_folder` 中真实存在的图像与对应 annotations。
	"""

	with open(source_ann_file, "r", encoding="utf-8") as f:
		ann_data = json.load(f)

	images = ann_data.get("images", [])
	annotations = ann_data.get("annotations", [])

	filtered_images = []
	valid_image_ids = set()

	for image_info in images:
		file_name = str(image_info.get("file_name", ""))
		candidates = []

		def add_candidate(name: str):
			if name and name not in candidates:
				candidates.append(name)

		add_candidate(file_name)
		root, ext = os.path.splitext(file_name)
		if ext.lower() in {".tif", ".tiff"}:
			add_candidate(root + ".png")

		if "_RGB_" in file_name:
			replaced = file_name.replace("_RGB_", "_")
			add_candidate(replaced)
			r2, e2 = os.path.splitext(replaced)
			if e2.lower() in {".tif", ".tiff"}:
				add_candidate(r2 + ".png")

		selected_name = None
		for cand in candidates:
			if os.path.isfile(os.path.join(val_img_folder, cand)):
				selected_name = cand
				break

		if selected_name is None:
			continue

		new_info = dict(image_info)
		new_info["file_name"] = selected_name
		filtered_images.append(new_info)
		valid_image_ids.add(int(new_info["id"]))

	filtered_annotations = [
		ann for ann in annotations if int(ann.get("image_id", -1)) in valid_image_ids
	]

	new_data = dict(ann_data)
	new_data["images"] = filtered_images
	new_data["annotations"] = filtered_annotations

	os.makedirs(os.path.dirname(out_ann_file), exist_ok=True)
	with open(out_ann_file, "w", encoding="utf-8") as f:
		json.dump(new_data, f, ensure_ascii=False, indent=2)

	return out_ann_file, len(filtered_images), len(images)


def run_evaluation(
	config_path: str,
	num_images: int,
	output_dir: str,
	base_ckpt: Optional[str],
	finetune_ckpt: Optional[str],
	prefer_isolated_val: bool,
	isolated_val_root: Optional[str],
	device_str: str,
	pred_threshold: float,
	label_prompt_overrides: Optional[List[str]],
	label_threshold_overrides: Optional[List[str]],
	suggest_label_thresholds: bool,
	threshold_candidates_text: str,
	combo_top_k: int,
	combo_top_n_per_label: int,
):
	if not (0.0 < pred_threshold < 1.0):
		raise ValueError(f"pred_threshold 必须在 (0,1) 区间内，当前为: {pred_threshold}")

	label_prompts = _parse_label_prompt_overrides(
		overrides=label_prompt_overrides,
		base_prompts=DEFAULT_LABEL_PROMPTS,
	)
	prompt_to_category = _build_prompt_to_category(label_prompts)
	label_thresholds = _parse_label_threshold_overrides(
		overrides=label_threshold_overrides,
		default_threshold=pred_threshold,
	)
	threshold_candidates = _parse_threshold_candidates(threshold_candidates_text)

	cfg = OmegaConf.load(config_path)

	device = torch.device(device_str)

	# 默认路径：从 config 中推断
	if base_ckpt is None:
		base_ckpt = cfg.paths.checkpoint_path

	if finetune_ckpt is None:
		ckpt_dir = cfg.trainer.checkpoint.save_dir
		finetune_ckpt = _auto_select_finetune_ckpt(ckpt_dir)
	if finetune_ckpt is None or not os.path.isfile(finetune_ckpt):
		raise FileNotFoundError(f"未找到 finetune checkpoint: {finetune_ckpt}")

	if output_dir is None:
		output_dir = os.path.join(cfg.paths.experiment_log_dir, "visualization")

	# 构建数据集与 collate_fn（复用训练时的 val pipeline）
	val_dataset_cfg = cfg.trainer.data.val.dataset
	collate_cfg = cfg.trainer.data.val.collate_fn

	# 为保证 exp1/exp2/exp3 公平对比，强制使用统一 test set: Golden_set/val
	if not prefer_isolated_val:
		print("警告: 为公平比较，已忽略 --disable-isolated-val，强制使用统一 test set。")

	if isolated_val_root is None:
		isolated_val_root = "/home/projectx/Sam3_data_engine/Golden_set/val"

	isolated_img_folder = os.path.join(isolated_val_root, "rgb")
	isolated_ann_file = os.path.join(isolated_val_root, "segmentation_val.json")

	if not os.path.isdir(isolated_img_folder) or not os.path.isfile(isolated_ann_file):
		raise FileNotFoundError(
			"统一 test set 缺失，无法做公平比较: "
			f"{isolated_img_folder} / {isolated_ann_file}"
		)

	mapped_ann_file = os.path.join(
		output_dir,
		"_cache",
		"segmentation_val_isolated_mapped.json",
	)
	mapped_ann_file, valid_count, total_count = _build_isolated_val_annotation(
		source_ann_file=isolated_ann_file,
		val_img_folder=isolated_img_folder,
		out_ann_file=mapped_ann_file,
	)

	if valid_count <= 0:
		raise RuntimeError(
			f"统一 test set 无可用图像: {isolated_img_folder} / {isolated_ann_file}"
		)

	val_dataset_cfg.img_folder = isolated_img_folder
	val_dataset_cfg.ann_file = mapped_ann_file
	print(
		"使用统一 test set: "
		f"{isolated_img_folder}, 匹配样本 {valid_count}/{total_count}"
	)

	# 覆盖 COCO loader 的 prompts 和 category_chunk_size，保证 5 个 text prompt 都被使用
	if hasattr(val_dataset_cfg, "coco_json_loader"):
		loader_cfg = val_dataset_cfg.coco_json_loader
		prompt_items = [
			{"id": int(cat_id), "name": name}
			for cat_id, name in sorted(label_prompts.items(), key=lambda x: x[0])
		]
		# 每个 datapoint 同时包含全部配置类别
		loader_cfg.category_chunk_size = len(prompt_items)
		loader_cfg.prompts = str(prompt_items)

	print("当前 label->prompt 配置:")
	for cat_id, prompt_name in sorted(label_prompts.items(), key=lambda x: x[0]):
		print(f"  label {cat_id}: '{prompt_name}'")

	val_dataset = instantiate(val_dataset_cfg)
	collate_fn = instantiate(collate_cfg)

	# 构建模型
	base_model, finetune_model = _build_models(
		cfg, device=device, base_ckpt=base_ckpt, finetune_ckpt=finetune_ckpt
	)

	total = len(val_dataset)
	num_images = min(num_images, total)
	indices = list(range(total))[:num_images]

	print(f"将从验证集前 {num_images} 张图像生成三图对比，可视化保存到: {output_dir}")
	print(f"预测二值化阈值 pred_threshold={pred_threshold:.3f}")
	if label_thresholds:
		print("当前 label->threshold 覆盖:")
		for cat_id, thr in sorted(label_thresholds.items(), key=lambda x: x[0]):
			print(f"  label {cat_id}: {thr:.3f}")

	os.makedirs(output_dir, exist_ok=True)
	base_test_losses: List[float] = []
	finetune_test_losses: List[float] = []

	label_iou_sum: Dict[int, Dict[float, float]] = {}
	label_iou_cnt: Dict[int, Dict[float, int]] = {}
	if suggest_label_thresholds:
		for cat_id in sorted(label_prompts.keys()):
			label_iou_sum[cat_id] = {thr: 0.0 for thr in threshold_candidates}
			label_iou_cnt[cat_id] = {thr: 0 for thr in threshold_candidates}

	for idx_pos, idx in enumerate(indices):
		datapoint = val_dataset[idx]
		batch_dict = collate_fn([datapoint])
		# dict_key 在 config 中为 "golden_set"
		(key, batched_dp), = batch_dict.items()

		# 保存原图（PIL）用于可视化
		raw_image = batched_dp.raw_images[0]
		w, h = raw_image.size

		# 按类别构造 GT mask 字典，并 resize 到原图大小
		stage_target = batched_dp.find_targets[0]
		stage_meta = batched_dp.find_metadatas[0]
		gt_cat_masks_small = _compute_category_masks(stage_target, stage_meta)
		gt_cat_masks = _resize_category_masks(gt_cat_masks_small, h, w)

		# 将 batch 拷贝到设备上做前向
		batch_on_device = copy_data_to_device(batched_dp, device)

		with torch.no_grad():
			base_out = base_model(batch_on_device)
			finetune_out = finetune_model(batch_on_device)

		base_stage = base_out[0]
		finetune_stage = finetune_out[0]
		base_test_loss = _compute_test_loss(base_stage, stage_target)
		finetune_test_loss = _compute_test_loss(finetune_stage, stage_target)
		if torch.isfinite(torch.tensor(base_test_loss)):
			base_test_losses.append(base_test_loss)
		if torch.isfinite(torch.tensor(finetune_test_loss)):
			finetune_test_losses.append(finetune_test_loss)

		# 按类别构造预测 mask，并 resize 到原图大小
		base_cat_masks_small = _compute_pred_category_masks(
			base_stage,
			batched_dp,
			prompt_to_category=prompt_to_category,
			label_thresholds=label_thresholds,
			pred_threshold=pred_threshold,
		)
		finetune_cat_masks_small = _compute_pred_category_masks(
			finetune_stage,
			batched_dp,
			prompt_to_category=prompt_to_category,
			label_thresholds=label_thresholds,
			pred_threshold=pred_threshold,
		)
		base_cat_masks = _resize_category_masks(base_cat_masks_small, h, w)
		finetune_cat_masks = _resize_category_masks(finetune_cat_masks_small, h, w)

		if suggest_label_thresholds:
			for thr in threshold_candidates:
				thr_cat_masks_small = _compute_pred_category_masks(
					finetune_stage,
					batched_dp,
					prompt_to_category=prompt_to_category,
					label_thresholds={},
					pred_threshold=thr,
				)
				thr_cat_masks = _resize_category_masks(thr_cat_masks_small, h, w)

				for cat_id in sorted(label_prompts.keys()):
					gt_mask = gt_cat_masks.get(cat_id)
					pred_mask = thr_cat_masks.get(cat_id)

					if gt_mask is None and pred_mask is None:
						continue
					fill_device = pred_mask.device if pred_mask is not None else gt_mask.device
					if gt_mask is None:
						gt_mask = torch.zeros((h, w), dtype=torch.bool, device=fill_device)
					if pred_mask is None:
						pred_mask = torch.zeros((h, w), dtype=torch.bool, device=fill_device)

					iou = _compute_iou(gt_mask, pred_mask)
					if iou is not None:
						label_iou_sum[cat_id][thr] += iou
						label_iou_cnt[cat_id][thr] += 1

		out_path = os.path.join(
			output_dir,
			f"sample_{idx_pos:02d}_idx_{idx}.png",
		)
		_plot_triplet(raw_image, gt_cat_masks, base_cat_masks, finetune_cat_masks, out_path)

		base_loss_str = f"{base_test_loss:.6f}" if torch.isfinite(torch.tensor(base_test_loss)) else "nan"
		finetune_loss_str = (
			f"{finetune_test_loss:.6f}"
			if torch.isfinite(torch.tensor(finetune_test_loss))
			else "nan"
		)
		print(
			f"  保存可视化: {out_path} | "
			f"test_loss(base,bce+dice)={base_loss_str}, "
			f"test_loss(finetune,bce+dice)={finetune_loss_str}"
		)

	if len(base_test_losses) > 0:
		base_avg = sum(base_test_losses) / len(base_test_losses)
		finetune_avg = sum(finetune_test_losses) / len(finetune_test_losses)
		print("\n=== Test Quantitative Summary (Unified Test Set) ===")
		print(f"test_set_root: {isolated_val_root}")
		print(f"num_samples: {len(base_test_losses)}")
		print(f"avg_test_loss/base: {base_avg:.6f}")
		print(f"avg_test_loss/finetune: {finetune_avg:.6f}")
		print(f"delta(finetune-base): {finetune_avg - base_avg:.6f}")

	if suggest_label_thresholds:
		print("\n=== Label Threshold Suggestion (Finetune IoU Sweep) ===")
		report = {
			"threshold_candidates": threshold_candidates,
			"label_prompt": {int(k): str(v) for k, v in sorted(label_prompts.items())},
			"recommended_threshold": {},
			"mean_iou": {},
			"combo_search": {
				"combo_top_n_per_label": int(combo_top_n_per_label),
				"combo_top_k": int(combo_top_k),
				"top_combinations": [],
			},
		}

		label_mean_iou: Dict[int, Dict[float, float]] = {}

		for cat_id in sorted(label_prompts.keys()):
			mean_iou_by_thr: Dict[float, float] = {}
			for thr in threshold_candidates:
				cnt = label_iou_cnt[cat_id][thr]
				if cnt > 0:
					mean_iou_by_thr[thr] = label_iou_sum[cat_id][thr] / cnt
			label_mean_iou[cat_id] = dict(mean_iou_by_thr)

			if not mean_iou_by_thr:
				print(f"label {cat_id} ({label_prompts[cat_id]}): 无有效样本，建议沿用 {pred_threshold:.3f}")
				report["recommended_threshold"][str(cat_id)] = pred_threshold
				report["mean_iou"][str(cat_id)] = {}
				continue

			best_thr, best_iou = max(mean_iou_by_thr.items(), key=lambda x: x[1])
			print(
				f"label {cat_id} ({label_prompts[cat_id]}): "
				f"recommend={best_thr:.3f}, mean_iou={best_iou:.4f}"
			)
			report["recommended_threshold"][str(cat_id)] = best_thr
			report["mean_iou"][str(cat_id)] = {
				f"{thr:.3f}": float(val) for thr, val in sorted(mean_iou_by_thr.items())
			}

		# 打印每个 label 的完整候选结果，便于对比。
		print("\n--- Per-label Candidate IoU Table ---")
		for cat_id in sorted(label_prompts.keys()):
			rows = []
			for thr in threshold_candidates:
				cnt = label_iou_cnt[cat_id][thr]
				if cnt <= 0:
					continue
				val = label_iou_sum[cat_id][thr] / cnt
				rows.append((thr, val))

			if not rows:
				print(f"label {cat_id} ({label_prompts[cat_id]}): no valid rows")
				continue

			rows_sorted = sorted(rows, key=lambda x: x[1], reverse=True)
			row_text = " | ".join([f"{thr:.2f}:{iou:.4f}" for thr, iou in rows_sorted])
			print(f"label {cat_id} ({label_prompts[cat_id]}): {row_text}")

		# 组合搜索（基于每类 mean IoU 的可分解估计分数）。
		combo_rankings = _rank_threshold_combinations(
			label_mean_iou=label_mean_iou,
			combo_top_n_per_label=combo_top_n_per_label,
			combo_top_k=combo_top_k,
		)
		report["combo_search"]["top_combinations"] = combo_rankings

		if combo_rankings:
			print("\n--- Top Threshold Combinations (estimated by per-label mean IoU) ---")
			for i, item in enumerate(combo_rankings, start=1):
				thr_map = item["thresholds"]
				thr_text = ", ".join(
					[
						f"{cat_id}:{float(thr_map[str(cat_id)]):.2f}"
						for cat_id in sorted(label_prompts.keys())
					]
				)
				print(
					f"#{i:02d} est_mean_iou={float(item['estimated_mean_iou']):.4f} | {thr_text}"
				)

			best_combo = combo_rankings[0]
			print("\n最终推荐组合(estimated):")
			print(
				", ".join(
					[
						f"label {cat_id}={float(best_combo['thresholds'][str(cat_id)]):.2f}"
						for cat_id in sorted(label_prompts.keys())
					]
				)
			)

		# 额外导出 CSV，方便在表格工具里看“不同组合”。
		candidate_csv_path = os.path.join(output_dir, "label_threshold_candidate_metrics.csv")
		with open(candidate_csv_path, "w", encoding="utf-8", newline="") as f:
			writer = csv.writer(f)
			writer.writerow(["label_id", "label_name", "threshold", "mean_iou", "num_samples"])
			for cat_id in sorted(label_prompts.keys()):
				for thr in threshold_candidates:
					cnt = label_iou_cnt[cat_id][thr]
					if cnt <= 0:
						continue
					mean_iou = label_iou_sum[cat_id][thr] / cnt
					writer.writerow([
						cat_id,
						label_prompts[cat_id],
						f"{thr:.3f}",
						f"{mean_iou:.6f}",
						cnt,
					])

		combo_csv_path = os.path.join(output_dir, "label_threshold_combo_ranking.csv")
		with open(combo_csv_path, "w", encoding="utf-8", newline="") as f:
			writer = csv.writer(f)
			headers = ["rank", "estimated_mean_iou"] + [f"label_{cat_id}_thr" for cat_id in sorted(label_prompts.keys())]
			writer.writerow(headers)
			for i, item in enumerate(combo_rankings, start=1):
				row = [i, f"{float(item['estimated_mean_iou']):.6f}"]
				for cat_id in sorted(label_prompts.keys()):
					row.append(f"{float(item['thresholds'][str(cat_id)]):.3f}")
				writer.writerow(row)

		report_path = os.path.join(output_dir, "label_threshold_suggestion.json")
		with open(report_path, "w", encoding="utf-8") as f:
			json.dump(report, f, ensure_ascii=False, indent=2)
		print(f"建议已保存: {report_path}")
		print(f"候选阈值明细已保存: {candidate_csv_path}")
		print(f"组合排名已保存: {combo_csv_path}")


def main():
	parser = argparse.ArgumentParser(
		description="对比 GT / 原始 SAM3 / finetune SAM3 的分割可视化 (Golden_set exp2)",
	)
	parser.add_argument(
		"--config",
		type=str,
		default="/home/projectx/Sam3_data_engine/experiments/exp2/config_resolved.yaml",
		help="训练时保存的 config_resolved.yaml 路径",
	)
	parser.add_argument(
		"--num-images",
		type=int,
		default=10,
		help="要可视化的图像数量",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default=None,
		help="输出可视化图片目录，默认为 experiments/exp2/visualization",
	)
	parser.add_argument(
		"--base-ckpt",
		type=str,
		default=None,
		help="原始 SAM3 checkpoint 路径（默认读 config.paths.checkpoint_path）",
	)
	parser.add_argument(
		"--finetune-ckpt",
		type=str,
		default=None,
		help=(
			"finetune checkpoint 路径（默认自动从 experiments/exp2/checkpoints 中选择 "
			"最大的 checkpoint_XX.pt，若不存在则使用 checkpoint.pt）"
		),
	)
	parser.add_argument(
		"--device",
		type=str,
		default="cuda" if torch.cuda.is_available() else "cpu",
		help="运行设备，例如 'cuda' 或 'cpu'",
	)
	parser.add_argument(
		"--isolated-val-root",
		type=str,
		default="/home/projectx/Sam3_data_engine/Golden_set/val",
		help="隔离验证集根目录（包含 rgb/ 和 segmentation_val.json）",
	)
	parser.add_argument(
		"--disable-isolated-val",
		action="store_true",
		help="禁用隔离验证集逻辑，改为使用 config 中的 val 路径",
	)
	parser.add_argument(
		"--pred-threshold",
		type=float,
		default=0.5,
		help="推理后处理二值化阈值，范围 (0,1)，例如 0.6/0.7",
	)
	parser.add_argument(
		"--label-prompt",
		action="append",
		default=None,
		help=(
			"按类别覆盖 prompt，可重复传入；格式 '<label_id>:<prompt>' 或 '<label_id>=<prompt>'，"
			"例如 --label-prompt 1:road_surface --label-prompt 2:river"
		),
	)
	parser.add_argument(
		"--label-threshold",
		action="append",
		default=None,
		help=(
			"按类别覆盖阈值，可重复传入；格式 '<label_id>:<threshold>' 或 '<label_id>=<threshold>'，"
			"例如 --label-threshold 1:0.65 --label-threshold 2:0.75"
		),
	)
	parser.add_argument(
		"--suggest-label-thresholds",
		action="store_true",
		help="基于 finetune 结果做每个 label 的阈值扫参并给出推荐值（按 IoU）",
	)
	parser.add_argument(
		"--threshold-candidates",
		type=str,
		default="0.5,0.55,0.6,0.65,0.7,0.75,0.8",
		help="阈值候选列表，逗号分隔，例如 '0.5,0.6,0.7'",
	)
	parser.add_argument(
		"--combo-top-n-per-label",
		type=int,
		default=5,
		help="组合排名时，每个 label 参与组合的候选阈值数量（按 mean IoU 前 N）",
	)
	parser.add_argument(
		"--combo-top-k",
		type=int,
		default=10,
		help="输出前 K 个阈值组合排名",
	)

	args = parser.parse_args()

	run_evaluation(
		config_path=args.config,
		num_images=args.num_images,
		output_dir=args.output_dir,
		base_ckpt=args.base_ckpt,
		finetune_ckpt=args.finetune_ckpt,
		prefer_isolated_val=not args.disable_isolated_val,
		isolated_val_root=args.isolated_val_root,
		device_str=args.device,
		pred_threshold=args.pred_threshold,
		label_prompt_overrides=args.label_prompt,
		label_threshold_overrides=args.label_threshold,
		suggest_label_thresholds=args.suggest_label_thresholds,
		threshold_candidates_text=args.threshold_candidates,
		combo_top_k=args.combo_top_k,
		combo_top_n_per_label=args.combo_top_n_per_label,
	)


if __name__ == "__main__":
	main()

