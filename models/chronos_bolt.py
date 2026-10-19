"""Chronos-Bolt Tiny (amazon/chronos-bolt-tiny) behind the learner's backbone interface.

Our RevIN normalises the context; Chronos-Bolt's own instance norm would then see mean 0 / std 1 data,
so it is skipped. blocks = [encoder, decoder]: the frozen prefix is the encoder output (the decoder reads
a single start token, so re-running it on every update is cheap); LoRA sits inside decoder blocks 2-3.
The head returns the 9 quantiles of the first `horizon` of its 64 steps.
"""
import torch
import torch.nn as nn

from models.mixer import PatchMixer

MEDIAN = 4


class _Encoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.stack = m.encoder

    def forward(self, h):
        mask = torch.ones(h.shape[:2], dtype=h.dtype, device=h.device)
        return self.stack(attention_mask=mask, inputs_embeds=h).last_hidden_state


class _Decoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.stack, self.start = m.decoder, m.config.decoder_start_token_id
        self.block = self.stack.block                       # so LoRA targets read "block.2.layer..."

    def forward(self, enc):
        ids = torch.full((enc.shape[0], 1), self.start, device=enc.device)
        mask = torch.ones(enc.shape[:2], dtype=enc.dtype, device=enc.device)
        return self.stack(input_ids=ids, encoder_hidden_states=enc, encoder_attention_mask=mask,
                          use_cache=False).last_hidden_state   # one decoding step: a cache is never reused


class ChronosBolt(nn.Module):
    revin = staticmethod(PatchMixer.revin)

    def __init__(self, repo, context=512, horizon=48, **_):
        super().__init__()
        from chronos import BaseChronosPipeline
        self.m = BaseChronosPipeline.from_pretrained(repo, device_map="cpu", torch_dtype=torch.float32).model
        cc = self.m.chronos_config
        assert cc.prediction_length >= horizon and context % cc.input_patch_size == 0
        self.context, self.horizon = context, horizon
        self.blocks = nn.ModuleList([_Encoder(self.m), _Decoder(self.m)])
        self.head = self.m.output_patch_embedding
        self._check_median()

    def _check_median(self):
        q = list(self.m.chronos_config.quantiles)
        assert len(q) > MEDIAN and q[MEDIAN] == 0.5, f"quantile {MEDIAN} is not the median: {q}"

    def embed(self, xn):
        p = self.m.patch(xn)                                        # (N, P, 16)
        e = self.m.input_patch_embedding(torch.cat([p, torch.ones_like(p)], dim=-1))
        reg = self.m.shared(torch.full((xn.shape[0], 1), self.m.config.reg_token_id, device=xn.device))
        return torch.cat([e, reg], dim=-2)

    def run_blocks(self, h, start=0, end=None):
        for blk in self.blocks[start:end]:
            h = blk(h)
        return h

    def head_out(self, h):
        q = self.head(h).view(h.shape[0], -1, self.m.chronos_config.prediction_length)
        return q[..., :self.horizon]

    def forward_norm(self, xn):
        return self.head_out(self.run_blocks(self.embed(xn)))[:, MEDIAN]
