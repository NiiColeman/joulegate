"""TTM-style patch mixer backbone (channel independent, RevIN normalised).

Kept deliberately close to Tiny Time Mixers: patching, alternating patch-mixing and
feature-mixing MLPs, and a flatten-linear forecasting head. About 0.6M parameters at
the default size. The model is split into prefix / suffix so the learner can cache the
frozen prefix activations (see engine/learner.py).
"""
import torch
import torch.nn as nn

CLIP = 10.0  # bound on RevIN-normalised targets; near-constant contexts make them explode


class MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class MixerBlock(nn.Module):
    def __init__(self, n_patches, d_model, expansion):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.patch_mlp = MLP(n_patches, expansion * n_patches)
        self.norm2 = nn.LayerNorm(d_model)
        self.feat_mlp = MLP(d_model, expansion * d_model)

    def forward(self, h):  # h: (B, P, d)
        h = h + self.patch_mlp(self.norm1(h).transpose(1, 2)).transpose(1, 2)
        return h + self.feat_mlp(self.norm2(h))


class PatchMixer(nn.Module):
    def __init__(self, context=256, horizon=24, patch_len=16, d_model=128, n_blocks=8, expansion=2):
        super().__init__()
        assert context % patch_len == 0
        self.context, self.horizon, self.patch_len = context, horizon, patch_len
        self.n_patches = context // patch_len
        self.embed_proj = nn.Linear(patch_len, d_model)
        self.blocks = nn.ModuleList(MixerBlock(self.n_patches, d_model, expansion) for _ in range(n_blocks))
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(self.n_patches * d_model, horizon)

    @staticmethod
    def revin(x, eps=1e-5):
        mu = x.mean(-1, keepdim=True)
        sd = x.std(-1, keepdim=True) + eps
        return (x - mu) / sd, mu, sd

    def embed(self, xn):
        return self.embed_proj(xn.reshape(xn.shape[0], self.n_patches, self.patch_len))

    def run_blocks(self, h, start=0, end=None):
        for blk in self.blocks[start:end]:
            h = blk(h)
        return h

    def head_out(self, h):
        return self.head(self.norm_out(h).flatten(1))

    def forward_norm(self, xn):
        return self.head_out(self.run_blocks(self.embed(xn)))

    def forward(self, x):
        xn, mu, sd = self.revin(x)
        return self.forward_norm(xn) * sd + mu
