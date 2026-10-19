"""timm ViT-Tiny with the backbone interface the learners use: embed, run_blocks, head_out.

Run at 96x96 (36 patches + the class token): timm resamples the pretrained position
embeddings when img_size differs from 224.
"""
import timm
import torch.nn as nn

VIT_TARGETS = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


class TinyViT(nn.Module):
    def __init__(self, name="vit_tiny_patch16_224.augreg_in21k_ft_in1k", img_size=96, num_classes=10,
                 pretrained=False):
        super().__init__()
        self.net = timm.create_model(name, pretrained=pretrained, img_size=img_size, num_classes=num_classes)
        self.blocks = self.net.blocks
        self.head = self.net.head

    def embed(self, x):
        n = self.net
        return n.norm_pre(n.patch_drop(n._pos_embed(n.patch_embed(x))))

    def run_blocks(self, h, start=0, end=None):
        for blk in self.blocks[start:end]:
            h = blk(h)
        return h

    def head_out(self, h):
        return self.net.forward_head(self.net.norm(h))

    def forward(self, x):
        return self.head_out(self.run_blocks(self.embed(x)))
