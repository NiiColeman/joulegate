"""Minimal LoRA for nn.Linear, applied to the last k mixer blocks.

A and B are plain nn.Linear modules so the energy counter sees them as ordinary layers.
"""
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_b.weight)
        self.scale = alpha / rank

    @property
    def weight(self):  # some wrapped models read it (e.g. T5's feed-forward checks wo.weight.dtype)
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scale


def add_lora(model, n_blocks, rank=4, alpha=8, targets=("feat_mlp.fc1", "feat_mlp.fc2"), train_head=False):
    """Freeze the model, wrap `targets` in the last `n_blocks` blocks, return trainable params.

    n_blocks = 0 means head-only fine-tuning (a common cheap baseline); train_head also trains
    the head when blocks are adapted (the vision learner's classifier).
    Returns (params, first_adapted_block_index).
    """
    for p in model.parameters():
        p.requires_grad_(False)
    total = len(model.blocks)
    first = total - n_blocks
    params = []
    for i in range(first, total):
        blk = model.blocks[i]
        for name in targets:
            parent_name, attr = name.rsplit(".", 1)
            parent = blk.get_submodule(parent_name)
            wrapped = LoRALinear(getattr(parent, attr), rank, alpha)
            setattr(parent, attr, wrapped)
            params += [wrapped.lora_a.weight, wrapped.lora_b.weight]
    if n_blocks == 0 or train_head:
        for p in model.head.parameters():
            p.requires_grad_(True)
            params.append(p)
    return params, first
