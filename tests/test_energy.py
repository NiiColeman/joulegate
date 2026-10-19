"""Checks that the analytic meter counts MACs exactly and prices only the gradient path.
Run: python -m tests.test_energy
"""
import torch

from engine.energy import AnalyticMeter
from engine.learner import Learner
from models.mixer import PatchMixer
from utils.config import build


def expected_forward_macs(m, n):
    P, d, e, pl, H = m.n_patches, m.blocks[0].feat_mlp.fc1.in_features, 2, m.patch_len, m.horizon
    per_block = d * (P * e * P * 2) + P * (d * e * d * 2)
    return n * (P * pl * d + len(m.blocks) * per_block + P * d * H)


def main():
    cfg = build("ettm1", "periodic", "analytic_45nm")
    torch.manual_seed(0)
    model = PatchMixer(**cfg["model"])
    n = 5
    meter = AnalyticMeter(model, cfg["energy"])
    x = torch.randn(n, cfg["model"]["context"])
    with torch.no_grad():
        meter.measure(lambda: model(x))
    exp = expected_forward_macs(model, n)
    assert meter.macs == exp, (meter.macs, exp)
    print(f"forward MACs exact: {meter.macs:,.0f}")

    # Deeper adapters must cost more per update: longer backward path, more stashing.
    costs = {}
    for k in (0, 1, 2, 4, 8):
        m = PatchMixer(**cfg["model"])
        lcfg = dict(cfg["learner"], adapt_blocks=k, cache_prefix=False)
        learner = Learner(m, lcfg)
        learner.meter = AnalyticMeter(m, cfg["energy"])
        xs = torch.randn(4, cfg["model"]["context"])
        for _ in range(4):
            learner.buffer.append((learner.model.revin(xs)[0], torch.randn(4, cfg["model"]["horizon"])))
        j, _, parts = learner.update()
        costs[k] = j
        print(f"adapt_blocks={k}: {j * 1e3:.3f} mJ  bwd-hooked MACs {learner.meter.macs:,.0f}")
    vals = [costs[k] for k in (0, 1, 2, 4, 8)]
    assert all(a < b for a, b in zip(vals, vals[1:])), vals

    # Caching the frozen prefix must give the same update at lower cost.
    out = {}
    for cache in (False, True):
        torch.manual_seed(0)
        m = PatchMixer(**cfg["model"])
        learner = Learner(m, dict(cfg["learner"], adapt_blocks=1, cache_prefix=cache))
        learner.meter = AnalyticMeter(m, cfg["energy"])
        torch.manual_seed(1)
        xs, ys = torch.randn(4, cfg["model"]["context"]), torch.randn(4, cfg["model"]["horizon"])
        with torch.no_grad():
            xn = m.revin(xs)[0]
            inp = m.run_blocks(m.embed(xn), 0, learner.first) if cache else xn
        for _ in range(4):
            learner.buffer.append((inp, ys))
        j, _, _ = learner.update()
        out[cache] = (j, torch.cat([p.detach().reshape(-1) for p in learner.params]))
    assert torch.allclose(out[False][1], out[True][1], atol=1e-6)
    assert out[True][0] < out[False][0]
    print(f"prefix cache: same update, {out[False][0] / out[True][0]:.1f}x cheaper")

    # Frozen layers below the first adapter must not appear in the backward count.
    m = PatchMixer(**cfg["model"])
    learner = Learner(m, dict(cfg["learner"], adapt_blocks=1))
    fired = []
    for name, mod in m.named_modules():
        if isinstance(mod, torch.nn.Linear):
            mod.register_full_backward_hook(lambda mod, gi, go, name=name: fired.append(name))
    x = torch.randn(3, cfg["model"]["context"])
    m.forward_norm(m.revin(x)[0]).sum().backward()
    assert all(f.startswith("blocks.7") or f == "head" for f in fired), fired
    print("backward path limited to adapted block + head:", sorted(set(fired)))
    print("all energy tests passed")


if __name__ == "__main__":
    main()
