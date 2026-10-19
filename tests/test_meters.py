import time

import pytest
import torch

from engine.energy import AnalyticMeter, SampledMeter, build_meter
from models.mixer import PatchMixer
from utils.config import build

FAKE = [{"name": "a", "type": "fake", "power_w": 10.0, "sample_s": 0.001},
        {"name": "b", "type": "fake", "power_w": 60.0, "quantum_j": 6.0, "sample_s": 0.01}]


@pytest.fixture
def sampled():
    m = build_meter({"backend": "sampled", "counters": FAKE, "max_seconds": 60, "pin": None}, model=None)
    yield m
    m.sampler.close()


def test_measure_records_nesting_and_windows(sampled):
    sampled.step = 3
    with sampled.window("decide") as outer:
        sampled.measure(lambda: time.sleep(0.01), "probe")
    ev = sampled.events()
    assert list(ev["kind"]) == ["decide", "probe"]
    assert list(ev["parent"]) == [-1, outer]
    assert list(ev["step"]) == [3, 3]
    assert ev["t1"][1] <= ev["t1"][0]
    assert sampled.last_id == outer


def test_provisional_joules_subtract_idle_power(sampled):
    sampled.p_idle = {"a": 10.0, "b": 0.0}
    _, j, sec, parts = sampled.measure(lambda: time.sleep(0.05), "update")
    assert parts == {"provisional": j}
    assert abs(j) < 60.0 * sec + 6.0          # a cancels exactly; b is at most one quantum off


def test_settled_waits_for_the_slow_counter(sampled):
    sampled.measure(lambda: None, "update")
    i = sampled.last_id
    assert sampled.settled(i) is None          # b refreshes only every 100 ms
    v = sampled.wait_settled(i, timeout=2.0)
    assert v == sampled.settled(i)


def test_measure_closes_the_window_when_fn_raises(sampled):
    with pytest.raises(ZeroDivisionError):
        sampled.measure(lambda: 1 / 0, "update")
    assert sampled.open == []
    assert sampled.events()["t1"][0] >= sampled.events()["t0"][0]


def test_analytic_meter_interface_is_unchanged_in_value():
    cfg = build("ettm1", "periodic", "analytic_45nm")
    model = PatchMixer(**cfg["model"])
    meter = AnalyticMeter(model, cfg["energy"])
    x = torch.randn(3, cfg["model"]["context"])
    with meter.window("decide") as i:
        assert i is None
    with torch.no_grad():
        _, j, _, _ = meter.measure(lambda: model(x), "infer")
    assert j > 0
    assert meter.settled(meter.last_id) == j == meter.wait_settled(meter.last_id)


def test_build_meter_rejects_unknown_backend():
    with pytest.raises(ValueError, match="rapl"):
        build_meter({"backend": "rapl"}, model=None)


def test_gpu_counters_are_bound_to_the_torch_device_by_uuid():
    """Under --exclusive NVML may see every GPU of the node; the counter must open the one torch uses."""
    from engine.energy import bind_gpu_counters
    specs = [{"name": "gpu", "type": "nvml", "index": 0, "require_single": True},
             {"name": "cpu_pkg", "type": "rapl", "path": "/x"}]
    out = bind_gpu_counters(specs, "GPU-1234")
    assert out[0]["uuid"] == "GPU-1234" and out[1] == specs[1]
    assert "uuid" not in specs[0]                                    # input left untouched
    assert bind_gpu_counters(specs, "1234")[0]["uuid"] == "GPU-1234"  # NVML wants the GPU- prefix


def test_analytic_meter_detaches_its_hooks():
    """A measured run prices one warm-up update analytically; its hooks must not linger in later windows."""
    cfg = build("ettm1", "periodic", "analytic_45nm")
    model = PatchMixer(**cfg["model"])
    meter = AnalyticMeter(model, cfg["energy"])
    meter.remove()
    x = torch.randn(3, cfg["model"]["context"])
    with torch.no_grad():
        meter.measure(lambda: model(x), "infer")
    assert meter.macs == 0


def test_analytic_meter_prices_attention_products_exactly():
    """softmax(QK^T)V is not a Linear layer; without this rule a ViT's attention would be free."""
    from timm.models.vision_transformer import Attention
    cfg = build("ettm1", "periodic", "analytic_45nm")
    att = Attention(dim=192, num_heads=3)
    meter = AnalyticMeter(att, cfg["energy"])
    B, N, C = 2, 37, 192
    with torch.no_grad():
        meter.measure(lambda: att(torch.randn(B, N, C)), "infer")
    linear = B * N * C * 3 * C + B * N * C * C               # qkv and proj
    products = 2 * B * 3 * N * N * 64                         # QK^T and PV
    assert meter.macs == linear + products


def test_attention_backward_is_priced_when_on_the_gradient_path():
    from timm.models.vision_transformer import Attention
    cfg = build("ettm1", "periodic", "analytic_45nm")
    att = Attention(dim=192, num_heads=3)
    meter = AnalyticMeter(att, cfg["energy"])
    B, N = 2, 37
    x = torch.randn(B, N, 192, requires_grad=True)
    meter.measure(lambda: att(x).sum().backward(), "update")
    linear = B * N * 192 * 3 * 192 + B * N * 192 * 192      # qkv and proj, forward
    P = B * 3 * N * N * 64
    # linear layers: forward + grad-input + grad-weight; products: QK^T, PV forward + dP, dV, dQ, dK
    assert meter.macs == 3 * linear + 2 * P + 4 * P


def test_analytic_meter_prices_a_convolution():
    cfg = build("ettm1", "periodic", "analytic_45nm")
    conv = torch.nn.Conv2d(3, 192, kernel_size=16, stride=16)
    for p in conv.parameters():
        p.requires_grad_(False)
    meter = AnalyticMeter(conv, cfg["energy"])
    with torch.no_grad():
        meter.measure(lambda: conv(torch.randn(2, 3, 96, 96)), "infer")
    assert meter.macs == 2 * 192 * 6 * 6 * 3 * 16 * 16


def test_t5_attention_products_are_priced_exactly():
    import os

    import torch
    from transformers import T5Config
    from transformers.models.t5.modeling_t5 import T5Attention

    from engine.energy import AnalyticMeter
    from utils.config import ROOT, load_yaml
    cfg = load_yaml(os.path.join(ROOT, "energy", "analytic_45nm.yaml"))
    att = T5Attention(T5Config(d_model=8, d_kv=4, num_heads=2), has_relative_attention_bias=False)
    meter = AnalyticMeter(att, cfg)
    x, kv = torch.randn(3, 5, 8), torch.randn(3, 7, 8)
    linear = 3 * 5 * 8 * 8 + 2 * 3 * 7 * 8 * 8 + 3 * 5 * 8 * 8          # q on x; k, v on kv; o on the output
    meter.active = True
    att(x, key_value_states=kv, query_length=5)   # transformers 4.51 needs it when called standalone
    assert meter.macs == linear + 2 * 3 * 2 * 5 * 7 * 4
