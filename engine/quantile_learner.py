"""Learner for quantile forecasters (Chronos-Bolt): the median is the point forecast that MASE scores
and buffers compare; updates minimise Chronos-Bolt's pinball loss over its quantiles."""
import torch

from engine.learner import Learner
from models.chronos_bolt import MEDIAN


class QuantileLearner(Learner):
    def __init__(self, model, cfg, meter=None, device="cpu"):
        super().__init__(model, cfg, meter, device)
        self.levels = torch.tensor(model.m.chronos_config.quantiles, device=self.dev).view(1, -1, 1)

    def _forecast(self, ctx):
        with torch.no_grad():
            x = torch.from_numpy(ctx).to(self.dev)
            xn, mu, sd = self.model.revin(x)
            h = self.model.run_blocks(self.model.embed(xn), 0, self.first)
            self._last_prefix = h.clone()
            yn = self.model.head_out(self.model.run_blocks(h, self.first))[:, MEDIAN]
            self.last_mu, self.last_sd, self._last_pred_n = mu, sd, yn
            return (yn * sd + mu).cpu().numpy(), xn

    def buffer_loss(self):
        """Normalised MSE of the median on the buffer, as for point forecasters (updates still minimise the
        pinball loss), so buffer_eval logs compare across backbones. No update; not metered."""
        inp, tgt = self._batch(replay=False)
        with torch.no_grad():
            h = inp if self.cache_prefix else self.model.run_blocks(self.model.embed(inp), 0, self.first)
            med = self.model.head_out(self.model.run_blocks(h, self.first))[:, MEDIAN]
            return float(torch.mean((med - tgt) ** 2))

    def _loss(self, q, tgt):
        t = tgt.unsqueeze(1)
        return (2 * torch.abs((t - q) * ((t <= q).float() - self.levels))).mean(-1).sum(-1).mean()
