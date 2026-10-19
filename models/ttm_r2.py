"""Granite TTM-R2 (ibm-granite/granite-timeseries-ttm-r2) behind the learner's backbone interface.

Rows are independent univariate series (N, L), as for PatchMixer. Our RevIN normalises, so the model
is loaded with scaling=None. The frozen prefix is the decoder input (backbone output, adapter and raw
patch residual); the adaptable blocks are the 2 decoder mixer layers; the head stays frozen.
"""
import torch
import torch.nn as nn

from models.mixer import PatchMixer
from models.vendor.ttm.modeling_tinytimemixer import TinyTimeMixerForPrediction


class TTMR2(nn.Module):
    revin = staticmethod(PatchMixer.revin)

    def __init__(self, repo, revision, context=512, horizon=48, freq_token=5, **_):
        super().__init__()
        self.m = TinyTimeMixerForPrediction.from_pretrained(repo, revision=revision, scaling=None)
        assert self.m.config.context_length == context and self.m.config.prediction_length == horizon
        self.context, self.horizon, self.freq_token = context, horizon, freq_token
        self.blocks = self.m.decoder.decoder_block.mixers
        self.head = self.m.head

    def embed(self, xn):
        out = self.m.backbone(past_values=xn.unsqueeze(-1),
                              freq_token=torch.full((xn.shape[0],), self.freq_token, device=xn.device))
        h, d = out.last_hidden_state, self.m.decoder
        h = d.adapter(h) if d.adapter is not None else h
        if d.decoder_raw_residual:
            p = out.patch_input
            if d.resolution_prefix_tuning and p.shape[-2] == h.shape[-2] - 1:
                p = torch.cat([torch.zeros_like(p[..., :1, :]), p], dim=-2)
            h = h + d.decoder_raw_embedding(p)
        return h

    def run_blocks(self, h, start=0, end=None):
        for blk in self.blocks[start:end]:
            h = blk(h)
        return h

    def head_out(self, h):
        return self.head(h, past_values=None)[..., 0]   # (N, H, 1) -> (N, H); past_values only feeds channel mixing (off)

    def forward_norm(self, xn):
        return self.head_out(self.run_blocks(self.embed(xn)))
