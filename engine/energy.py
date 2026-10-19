"""Energy meters. Both expose measure(fn, kind) -> (result, joules, seconds, parts),
window(kind), settled(i) and wait_settled(i).

- analytic : counts MACs, elementwise ops and bytes moved per layer via forward and
             backward hooks, then prices them with per-operation energies (defaults:
             Horowitz, ISSCC 2014, 45 nm). Backward hooks only fire for layers on the
             gradient path, so backward-path depth and activation stashing are priced
             automatically. Deterministic and hardware independent, but a model.
             Values are final immediately; window() records nothing.
- sampled  : hardware counters (RAPL, NVML, a sysfs power rail) polled by a Sampler
             process (engine/power.py). measure() and window() record event windows;
             the joules measure() returns are provisional until the counters refresh
             past the window (settled). Final values come from engine/attribution.py.
"""
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

from engine.power import Sampler

class AnalyticMeter:
    def __init__(self, model, cfg):
        self.c = cfg
        self.b = cfg["bytes_per_elem"]
        self.active = False
        self._reset()
        self.values, self.last_id, self.step = [], -1, -1
        self.hooks = []
        for m in model.modules():
            if self._is_t5_attention(m):  # its query/key lengths come from keyword arguments
                self.hooks.append(m.register_forward_hook(self._fwd_t5, with_kwargs=True))
                self.hooks.append(m.register_full_backward_hook(self._bwd_t5))
            elif (isinstance(m, (nn.Linear, nn.LayerNorm, nn.GELU, nn.ReLU, nn.Conv2d)) or self._is_attention(m)
                  or type(m).__name__ == "T5LayerNorm"):
                self.hooks.append(m.register_forward_hook(self._fwd))
                self.hooks.append(m.register_full_backward_hook(self._bwd))

    def remove(self):
        """Detach every hook, e.g. after pricing one update analytically inside a measured run."""
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def _reset(self):
        self.macs = self.ops = self.sram = self.dram = 0.0

    @staticmethod
    def _is_attention(m):
        """timm-style attention: its QK^T and PV products are not modules, so price them here."""
        return all(hasattr(m, a) for a in ("num_heads", "head_dim", "qkv"))

    @staticmethod
    def _is_t5_attention(m):
        """Hugging Face T5 attention (Chronos-Bolt): QK^T and PV are not modules either."""
        return all(hasattr(m, a) for a in ("n_heads", "key_value_proj_dim", "q", "o"))

    def _t5_products(self, m):
        B, nq, nk = m._jg_shape
        s = B * m.n_heads * nq * nk
        return s * m.key_value_proj_dim, s

    def _fwd_t5(self, m, args, kwargs, out):
        if not self.active:
            return
        x = args[0] if args else kwargs["hidden_states"]
        kv = kwargs.get("key_value_states")
        src = kv if kv is not None else x
        m._jg_shape = (x.shape[0], x.shape[1], src.shape[1])
        trains = lambda lin: any(p.requires_grad for p in lin.parameters())
        # backward products autograd needs: dQ if the query side carries a gradient, dK / dV if the key / value side does
        m._jg_need = (x.requires_grad or trains(m.q), src.requires_grad or trains(m.k), src.requires_grad or trains(m.v))
        macs, scores = self._t5_products(m)
        self.macs += 2 * macs                                       # QK^T and PV
        self.ops += self.c["elementwise_ops"] * scores              # softmax
        self._move(2 * scores * self.b)

    def _bwd_t5(self, m, grad_in, grad_out):
        if not self.active or not hasattr(m, "_jg_shape"):
            return
        macs, scores = self._t5_products(m)
        self.macs += (1 + sum(m._jg_need)) * macs                   # dP, plus dQ / dK / dV where needed
        self.ops += self.c["elementwise_ops_bwd"] * scores
        self._move(4 * scores * self.b)

    def _attention_products(self, m, x):
        B, N = x.shape[0], x.shape[1]
        return B * m.num_heads * N * N * m.head_dim, B * m.num_heads * N * N

    def _move(self, nbytes):
        if nbytes <= self.c["onchip_bytes"]:
            self.sram += nbytes
        else:
            self.dram += nbytes

    def _fwd(self, m, inp, out):
        if not self.active:
            return
        x = inp[0]
        if self._is_attention(m):
            macs, scores = self._attention_products(m, x)
            self.macs += 2 * macs                                   # QK^T and PV
            self.ops += self.c["elementwise_ops"] * scores          # softmax
            self._move(2 * scores * self.b)                         # score matrix written and read
            return
        if isinstance(m, nn.Conv2d):
            kh, kw = m.kernel_size
            self.macs += out.numel() * (m.in_channels // m.groups) * kh * kw
            self._move(m.weight.numel() * self.b)
            self._move(x.numel() * self.b)
            self._move(out.numel() * self.b)
            return
        if isinstance(m, nn.Linear):
            rows = x.numel() // m.in_features
            self.macs += rows * m.in_features * m.out_features
            self._move(m.weight.numel() * self.b)
            self._move(x.numel() * self.b)
            self._move(out.numel() * self.b)
        else:
            self.ops += self.c["elementwise_ops"] * x.numel()
            self._move(x.numel() * self.b)
            self._move(out.numel() * self.b)

    def _bwd(self, m, grad_in, grad_out):
        if not self.active:
            return
        g = grad_out[0]
        if self._is_attention(m):
            macs, scores = self._attention_products(m, g)
            self.macs += 4 * macs                                   # dP, dV, dQ, dK
            self.ops += self.c["elementwise_ops_bwd"] * scores
            self._move(4 * scores * self.b)
            return
        if isinstance(m, nn.Conv2d):
            kh, kw = m.kernel_size
            per = g.numel() * (m.in_channels // m.groups) * kh * kw
            if grad_in[0] is not None:
                self.macs += per
            if m.weight.requires_grad:
                self.macs += per
            self._move(2 * g.numel() * self.b)
            return
        if isinstance(m, nn.Linear):
            rows = g.numel() // m.out_features
            n_in = rows * m.in_features
            if grad_in[0] is not None:  # propagate the gradient further down
                self.macs += rows * m.in_features * m.out_features
                self._move(m.weight.numel() * self.b)
                self._move(g.numel() * self.b)
                self._move(n_in * self.b)
            if m.weight.requires_grad:  # weight gradient needs the stashed input
                self.macs += rows * m.in_features * m.out_features
                self._move(2 * n_in * self.b)  # stash write in forward + read here
                self._move(g.numel() * self.b)
                self._move(m.weight.numel() * self.b)
        elif grad_in[0] is not None:
            self.ops += self.c["elementwise_ops_bwd"] * g.numel()
            self._move(2 * g.numel() * self.b)  # stashed input write + read
            self._move(2 * g.numel() * self.b)  # grad_out read, grad_in write

    def optimizer_step(self, n_params):
        """Adam: read param, grad, m, v and write param, m, v."""
        self.ops += self.c["adam_ops_per_param"] * n_params
        self._move(7 * n_params * self.b)

    def store(self, nbytes):
        """Charge an explicit write, e.g. caching prefix activations."""
        self._move(nbytes)

    def joules(self):
        c = self.c
        parts = {
            "compute": (self.macs * c["mac_pj"] + self.ops * c["elementwise_pj"]) * 1e-12,
            "sram": self.sram * c["sram_pj_per_byte"] * 1e-12,
            "dram": self.dram * c["dram_pj_per_byte"] * 1e-12,
        }
        return sum(parts.values()), parts

    def measure(self, fn, kind=None):
        self._reset()
        self.active = True
        t0 = time.perf_counter()
        try:
            out = fn()
        finally:
            self.active = False
        sec = time.perf_counter() - t0
        joules, parts = self.joules()
        static = self.c.get("static_w", 0.0) * sec
        parts["static"] = static
        self.values.append(joules + static)
        self.last_id = len(self.values) - 1
        return out, joules + static, sec, parts

    @contextmanager
    def window(self, kind):
        yield None

    def settled(self, i):
        return self.values[i]

    def wait_settled(self, i, timeout=5.0):
        return self.values[i]


class SampledMeter:
    """Records event windows against a running Sampler."""

    def __init__(self, sampler, device="cpu"):
        self.sampler, self.cuda = sampler, device == "cuda"
        self.p_idle = {n: 0.0 for n in sampler.names}
        self.kind, self.parent, self.step_of, self.t0, self.t1, self.prov = [], [], [], [], [], []
        self.open, self.step, self.last_id = [], -1, -1

    def _sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    def _begin(self, kind):
        self._sync()
        i = len(self.kind)
        self.kind.append(kind)
        self.parent.append(self.open[-1] if self.open else -1)
        self.step_of.append(self.step)
        self.t1.append(np.nan)
        self.prov.append(np.nan)
        self.open.append(i)
        self.t0.append(time.perf_counter())
        return i

    def _end(self, i):
        self._sync()
        self.t1[i] = time.perf_counter()
        self.open.pop()
        self.last_id = i

    def _incr(self, i):
        e = self.sampler.energy(self.t0[i], self.t1[i])
        dur = self.t1[i] - self.t0[i]
        return sum(e[n] - self.p_idle[n] * dur for n in e)

    def measure(self, fn, kind=None):
        i = self._begin(kind)
        try:
            out = fn()
        finally:
            self._end(i)
        j = self._incr(i)
        self.prov[i] = j
        return out, j, self.t1[i] - self.t0[i], {"provisional": j}

    @contextmanager
    def window(self, kind):
        i = self._begin(kind)
        try:
            yield i
        finally:
            self._end(i)

    def settled(self, i):
        self.sampler.check()
        return self._incr(i) if self.sampler.settled_after(self.t1[i]) else None

    def wait_settled(self, i, timeout=5.0):
        deadline = time.perf_counter() + timeout
        while (v := self.settled(i)) is None:
            if time.perf_counter() > deadline:
                raise RuntimeError(f"energy of event {i} did not settle within {timeout} s")
            time.sleep(0.002)
        return v

    def optimizer_step(self, n_params):
        pass

    def store(self, nbytes):
        pass

    def events(self):
        return {"kind": np.array(self.kind), "parent": np.array(self.parent, dtype=np.int64),
                "step": np.array(self.step_of, dtype=np.int64), "t0": np.array(self.t0),
                "t1": np.array(self.t1), "provisional": np.array(self.prov)}


def bind_gpu_counters(counters, uuid):
    """Copies of the counter specs with every NVML counter bound to one GPU by UUID. Under
    --exclusive NVML can see every GPU of the node, and its index order need not match CUDA's."""
    uuid = uuid if uuid.startswith("GPU-") else "GPU-" + uuid
    return [dict(c, uuid=uuid) if c["type"] == "nvml" else c for c in counters]


def build_meter(cfg, model, device="cpu"):
    kind = cfg["backend"]
    if kind == "analytic":
        return AnalyticMeter(model, cfg)
    if kind == "sampled":
        counters = cfg["counters"]
        if device == "cuda":
            counters = bind_gpu_counters(counters, str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid))
        sampler = Sampler(counters, cfg.get("max_seconds", 3600), (cfg.get("pin") or {}).get("sampler"),
                          cfg.get("start_timeout_s", 120.0))
        return SampledMeter(sampler, device)
    raise ValueError(f"unknown energy backend {kind!r} (use analytic or sampled)")
