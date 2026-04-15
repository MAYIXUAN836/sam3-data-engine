"""
SAM3 LoRA 注入脚本
==================
在 SAM3 官方训练框架启动前，把 LoRA 注入到 Vision Backbone 的 qkv 层

策略 (Loop 1):
  - Vision Backbone qkv: 注入 LoRA r=16
  - Vision Backbone 其他层: 冻结
  - Language Backbone: 完全冻结
  - Mask Decoder / 其他: 全量微调

用法:
  在训练脚本里 import 这个模块，调用 apply_lora_to_sam3(model)

  或者直接运行验证:
  python lora_injection.py
"""

import math
import torch
import torch.nn as nn
from pathlib import Path


class LoRALinear(nn.Module):
    """
    在已有 Linear 层上注入 LoRA 旁路
    forward: output = original(x) + (x @ A.T) @ B.T * scaling
    初始化: A 随机小值，B 零矩阵 → 初始输出不变，不破坏预训练权重
    """
    def __init__(self, original: nn.Linear, r: int = 16, alpha: float = 16.0):
        super().__init__()
        self.original = original
        self.r        = r
        self.scaling  = alpha / r
        in_dim        = original.in_features
        out_dim       = original.out_features

        # LoRA 参数
        self.lora_A = nn.Parameter(torch.empty(r, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, r))

        # 初始化 A（kaiming uniform，标准做法）
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # 冻结原始权重
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

    def forward(self, x):
        orig_out = self.original(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return orig_out + lora_out * self.scaling

    def __repr__(self):
        return (f"LoRALinear(in={self.original.in_features}, "
                f"out={self.original.out_features}, r={self.r})")


def apply_lora_to_sam3(model: nn.Module, r: int = 16, alpha: float = 16.0):
    """
    把 LoRA 注入到 SAM3 Vision Backbone 的所有 qkv 层
    同时设置好冻结策略

    返回: (lora_params_count, total_params_count)
    """

    # ── Step 1: 全部冻结 ──────────────────────────────────
    for p in model.parameters():
        p.requires_grad_(False)

    # ── Step 2: 注入 LoRA 到 Vision Backbone qkv ─────────
    trunk  = model.backbone.vision_backbone.trunk
    blocks = trunk.blocks
    n_injected = 0

    for i, block in enumerate(blocks):
        if hasattr(block, 'attn') and hasattr(block.attn, 'qkv'):
            orig_qkv = block.attn.qkv
            if isinstance(orig_qkv, nn.Linear):
                block.attn.qkv = LoRALinear(orig_qkv, r=r, alpha=alpha)
                n_injected += 1

    print(f"  LoRA 注入: {n_injected} 个 qkv 层 (r={r}, alpha={alpha})")

    # ── Step 3: 解冻 Mask Decoder ─────────────────────────
    # SAM3 的 decoder 相关模块
    decoder_keywords = [
        'sam_mask_decoder', 'mask_decoder',
        'transformer', 'decoder',
        'query_feat', 'query_pos',
    ]
    decoder_unfrozen = 0
    for name, module in model.named_modules():
        name_lower = name.lower()
        if any(kw in name_lower for kw in decoder_keywords):
            # 排除 vision/language backbone
            if 'vision_backbone' not in name and 'language_backbone' not in name:
                for p in module.parameters():
                    if not p.requires_grad:
                        p.requires_grad_(True)
                        decoder_unfrozen += 1

    # ── Step 4: 解冻 neck / FPN (如果有) ─────────────────
    for name, module in model.named_modules():
        if 'neck' in name.lower() or 'fpn' in name.lower():
            for p in module.parameters():
                p.requires_grad_(True)

    # ── Step 5: 解冻 LoRA 参数 ───────────────────────────
    lora_unfrozen = 0
    for name, p in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            p.requires_grad_(True)
            lora_unfrozen += 1

    # ── 统计 ─────────────────────────────────────────────
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  总参数量:   {total/1e6:.1f}M")
    print(f"  可训练参数: {trainable/1e6:.2f}M  ({100*trainable/total:.2f}%)")
    print(f"  LoRA 参数:  {lora_unfrozen} 个张量")

    return trainable, total


def patch_sam3_trainer():
    """
    Monkey-patch SAM3 的 Trainer，在模型加载后自动注入 LoRA
    在训练脚本最开头 import 这个函数并调用即可
    """
    try:
        from sam3.train import trainer as trainer_module
        original_setup = trainer_module.Trainer.setup

        def patched_setup(self, *args, **kwargs):
            result = original_setup(self, *args, **kwargs)
            print("\n[LoRA Injection] 开始注入 LoRA...")
            apply_lora_to_sam3(self.model, r=16, alpha=16.0)
            print("[LoRA Injection] 完成！\n")
            return result

        trainer_module.Trainer.setup = patched_setup
        print("[LoRA Injection] Trainer patch 成功")
        return True

    except Exception as e:
        print(f"[LoRA Injection] Patch 失败: {e}")
        return False


if __name__ == "__main__":
    # ── 验证模式：加载模型测试注入效果 ──────────────────
    print("=" * 55)
    print("  SAM3 LoRA 注入验证")
    print("=" * 55)

    from sam3.model_builder import build_sam3_image_model

    print("\n加载模型...")
    model = build_sam3_image_model(
        bpe_path="/home/projectx/Sam3_data_engine/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        device="cpu",
        checkpoint_path="/home/projectx/Sam3_data_engine/checkpoints/sam3.pt",
    )

    print("\n注入 LoRA...")
    trainable, total = apply_lora_to_sam3(model, r=16, alpha=16.0)

    print("\n验证 qkv 层类型:")
    blocks = model.backbone.vision_backbone.trunk.blocks
    for i in [0, 1, 15, 31]:
        qkv = blocks[i].attn.qkv
        print(f"  block[{i:2d}].attn.qkv: {type(qkv).__name__}")

    print("\n验证梯度状态 (前5个可训练参数):")
    count = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"  ✅ {name}  shape={list(p.shape)}")
            count += 1
            if count >= 5:
                print("  ...")
                break

    print(f"\n✅ LoRA 注入验证完成！")
    print(f"   参数效率: {100*trainable/total:.2f}% 参与训练")