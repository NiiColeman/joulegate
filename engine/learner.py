"""Online learner: frozen backbone + LoRA adapters, a FIFO buffer of recent labelled
windows, and metered forecast / update calls. Policies never touch this file.

Updates are full-batch over the buffer, so they are deterministic given the buffer,
which lets the oracle policy preview an update exactly.
"""
import copy
import warnings
from collections import deque

import numpy as np
import torch

from models.lora import add_lora
from models.mixer import CLIP

# The analytic meter uses full backward hooks on layers whose inputs need no gradient.
warnings.filterwarnings("ignore", message="Full backward hook is firing")


class Learner:
    def __init__(self, model, cfg, meter=None, device="cpu"):
        self.model, self.meter, self.cfg = model, meter, cfg
        self.dev = torch.device(device)
        targets = tuple(cfg.get("lora_targets", ("feat_mlp.fc1", "feat_mlp.fc2")))
        self.params, self.first = add_lora(model, cfg["adapt_blocks"], cfg["rank"], cfg["alpha"], targets,
                                           cfg.get("train_head", False))
        self.model.to(self.dev)  # moves parameters in place, so self.params stay valid
        self.n_params = sum(p.numel() for p in self.params)
        self.opt = torch.optim.Adam(self.params, lr=cfg["lr"])
        self.cache_prefix = cfg["cache_prefix"]
        self.buffer = deque(maxlen=cfg["buffer_steps"])
        self._last_prefix = None
        # Optional replay memory (Prabhu et al. 2023's naive baseline): a reservoir over every labelled
        # step; an update draws `frac` of its batch from it and the rest from the newest FIFO items.
        self.replay = cfg.get("replay")
        self.memory, self.seen = [], 0
        self.rng = np.random.default_rng(self.replay.get("seed", 0)) if self.replay else None
        self._base = None
        self.model.eval()

    # ------------------------------------------------------------------ forecasting
    def _forecast(self, ctx):
        with torch.no_grad():
            x = torch.from_numpy(ctx).to(self.dev)
            xn, mu, sd = self.model.revin(x)
            h = self.model.run_blocks(self.model.embed(xn), 0, self.first)
            self._last_prefix = h.clone()
            yn = self.model.head_out(self.model.run_blocks(h, self.first))
            # kept for cheap policies (input-shift gate, residual VoU): already computed, no extra work
            self.last_mu, self.last_sd, self._last_pred_n = mu, sd, yn
            return (yn * sd + mu).cpu().numpy(), xn

    def predict(self, ctx):
        (pred, xn), joules, sec, _ = self.meter.measure(lambda: self._forecast(ctx), "infer")
        self._last_xn = xn
        return pred, joules, sec

    def push(self, ctx, y):
        """Store the newest labelled window. Returns the energy charged for the write."""
        x = torch.from_numpy(ctx).to(self.dev)
        _, mu, sd = self.model.revin(x)
        yn = ((torch.from_numpy(y).to(self.dev) - mu) / sd).clamp(-CLIP, CLIP)
        self._last_target_n = yn
        item = (self._last_prefix if self.cache_prefix else self._last_xn, yn)
        nbytes = (item[0].numel() + yn.numel()) * 4
        self.buffer.append(item)
        nbytes += self._remember(item, nbytes)
        _, joules, _, _ = self.meter.measure(lambda: self.meter.store(nbytes), "store")
        return joules

    def _remember(self, item, nbytes):
        """Reservoir sampling into the replay memory. Returns the bytes written (0 if not kept)."""
        if not self.replay:
            return 0
        self.seen += 1
        if len(self.memory) < self.replay["size"]:
            self.memory.append(item)
            return nbytes
        k = int(self.rng.integers(self.seen))
        if k < self.replay["size"]:
            self.memory[k] = item
            return nbytes
        return 0

    def _items(self, replay=True):
        """The update batch's items: the FIFO buffer, or with replay its newest items plus memory draws
        (same count, so an update costs the same with or without replay)."""
        items = list(self.buffer)
        if not (replay and self.replay and self.memory):
            return items
        m = min(int(round(self.replay["frac"] * len(items))), len(self.memory))
        idx = self.rng.choice(len(self.memory), m, replace=False)
        return items[m:] + [self.memory[k] for k in idx]

    # ------------------------------------------------------------------ adaptation
    def _batch(self, replay=True):
        items = self._items(replay)
        inp = torch.cat([b[0] for b in items])
        tgt = torch.cat([b[1] for b in items])
        return inp, tgt

    def _update(self):
        inp, tgt = self._batch()
        loss = None
        for _ in range(self.cfg["steps_per_update"]):
            h = inp if self.cache_prefix else self.model.run_blocks(self.model.embed(inp), 0, self.first)
            pred = self.model.head_out(self.model.run_blocks(h, self.first))
            loss = self._loss(pred, tgt)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            self.meter.optimizer_step(self.n_params)
        return float(loss.detach())

    def _loss(self, pred, tgt):
        """Training loss; the vision learner overrides it with cross-entropy."""
        return torch.mean((pred - tgt) ** 2)

    def update(self, kind="update"):
        loss, joules, sec, parts = self.meter.measure(self._update, kind)
        return joules, sec, parts

    def _probe(self):
        inp, tgt = self.buffer[-1]
        h = inp if self.cache_prefix else self.model.run_blocks(self.model.embed(inp), 0, self.first)
        loss = self._loss(self.model.head_out(self.model.run_blocks(h, self.first)), tgt)
        grads = torch.autograd.grad(loss, self.params)
        return torch.cat([g.reshape(-1) for g in grads])

    def probe_grad(self):
        """Adapter gradient on the newest window only (no step). Returns (grad, joules)."""
        g, joules, _, _ = self.meter.measure(self._probe, "probe")
        return g, joules

    def set_base(self, load, *args):
        """Held-out data from the backbone's own training distribution (`load()` returns it; `args` are
        what `_base_items` needs), so adapter_loss(base=True) can score the pretrained model on it."""
        self._base, self._base_args = load, args

    def _base_rows(self):
        """A fixed sample (seed 0) of base rows the size of a full buffer."""
        n_all = len(self._base_labels())
        n = min(n_all, self.cfg["buffer_steps"] * self.buffer[-1][0].shape[0])
        return np.sort(np.random.default_rng(0).choice(n_all, n, replace=False))

    def _base_labels(self):
        return self._base()

    def _base_items(self, idx):
        """Pretraining windows as buffer items: (prefix or normalised context, normalised target)."""
        horizon, = self._base_args
        w = torch.from_numpy(self._base()[idx]).to(self.dev)
        xn, mu, sd = self.model.revin(w[:, :-horizon])
        yn = ((w[:, -horizon:] - mu) / sd).clamp(-CLIP, CLIP)
        return (self.model.run_blocks(self.model.embed(xn), 0, self.first) if self.cache_prefix else xn), yn

    def _adapter_loss(self, newest, base=False):
        if base:
            inp, tgt = self._base_items(self._base_rows())
        else:
            items = [self.buffer[-1]] if newest else self._items(replay=False)
            inp = torch.cat([b[0] for b in items])
            tgt = torch.cat([b[1] for b in items])
        with torch.no_grad():
            h = inp if self.cache_prefix else self.model.run_blocks(self.model.embed(inp), 0, self.first)
            return float(self._loss(self.model.head_out(self.model.run_blocks(h, self.first)), tgt))

    def adapter_loss(self, newest=False, base=False):
        """Training loss of the current adapter on the newest stored window, on the whole FIFO buffer (no
        update, no replay), or on a buffer-sized sample of base data (set_base). Metered as probe energy,
        including the base sample's forward through the frozen prefix. Returns (loss, joules)."""
        if base and self._base is None:
            raise RuntimeError("adapter_loss(base=True) needs set_base()")
        loss, joules, _, _ = self.meter.measure(lambda: self._adapter_loss(newest, base), "probe")
        return loss, joules

    def _item_losses(self, which):
        if which == "base":
            inp, tgt = self._base_items(self._base_rows())
            r = self.buffer[-1][0].shape[0]
            items = [(inp[i:i + r], tgt[i:i + r]) for i in range(0, len(inp), r)]
        else:
            items = [self.buffer[-1]] if which == "newest" else self._items(replay=False)
        losses, feats = [], []
        with torch.no_grad():
            for x, y in items:
                h = x if self.cache_prefix else self.model.run_blocks(self.model.embed(x), 0, self.first)
                losses.append(float(self._loss(self.model.head_out(self.model.run_blocks(h, self.first)), y)))
                feats.append(h.reshape(-1, h.shape[-1]).mean(0).float().cpu().numpy())
        return np.array(losses), np.stack(feats)

    def item_losses(self, which):
        """Per-item training loss of the current adapter, and each item's feature (its frozen-prefix
        activations averaged to one vector), on `which`: "newest" (the newest stored window), "buffer"
        (the FIFO buffer, no replay) or "base" (a buffer-sized sample of base data, set_base). Metered
        as probe energy. Returns (losses, features, joules)."""
        if which == "base" and self._base is None:
            raise RuntimeError("item_losses('base') needs set_base()")
        (losses, feats), joules, _, _ = self.meter.measure(lambda: self._item_losses(which), "probe")
        return losses, feats, joules

    # ------------------------------------------------------------------ snapshots
    def snapshot(self):
        return [p.detach().clone() for p in self.params], copy.deepcopy(self.opt.state_dict())

    def restore(self, snap):
        values, opt_state = snap
        with torch.no_grad():
            for p, v in zip(self.params, values):
                p.copy_(v)
        self.opt.load_state_dict(opt_state)

    def trial_update(self):
        """Measure one update without keeping it (cost calibration). Returns joules."""
        snap = self.snapshot()
        joules, _, _ = self.update(kind="calib")
        self.restore(snap)
        return joules

    def last_residual(self):
        """Newest forecast error in RevIN-normalised units, flattened over channels x horizon."""
        return (self._last_pred_n - self._last_target_n).reshape(-1)

    def buffer_loss(self):
        """Training loss of the current adapter on the buffer (normalised MSE for forecasting).
        No update; not metered."""
        inp, tgt = self._batch(replay=False)
        with torch.no_grad():
            h = inp if self.cache_prefix else self.model.run_blocks(self.model.embed(inp), 0, self.first)
            pred = self.model.head_out(self.model.run_blocks(h, self.first))
            return float(self._loss(pred, tgt))

    def eval_mse(self, windows, horizon, batch=512):
        """Normalised MSE on fixed windows (retention on the pretraining distribution)."""
        errs = []
        with torch.no_grad():
            for i in range(0, len(windows), batch):
                w = torch.from_numpy(windows[i:i + batch]).to(self.dev)
                xn, mu, sd = self.model.revin(w[:, :-horizon])
                yn = ((w[:, -horizon:] - mu) / sd).clamp(-CLIP, CLIP)
                errs.append(((self.model.forward_norm(xn) - yn) ** 2).mean(1))
        return float(torch.cat(errs).mean().cpu())
