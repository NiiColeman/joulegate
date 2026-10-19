"""Measured-energy run protocol.

    idle (idle_s) -> warm-up (booked as `other`) -> stream loop -> idle (idle_s)

The warm-up also measures the price scale: measured joules per full-buffer trial update over
the analytic joules of the same update. Policies that price updates in joules (VoU) divide
by it, so lam keeps the scale it was tuned on. A single update (~15 ms) is far below what
NVML (100 ms steps) or a shared socket can resolve, so trial updates run back to back in one
window of at least `scale_window_s`, and the window's energy is divided by their number.
Forecasts and VoU's gradient probes get the same treatment, so every run also records exact
joules per call for all three operations (`per_call` in the summary).

then attribute energy to events from the sampled counters, subtract the mean idle power
of the two idle windows, check closure, and build the outputs. The sampler is always
closed, and nothing is returned (so nothing is saved) unless every check passed.
"""
import os
import time

import numpy as np

from engine.attribution import ORDER, attribute, idle_power
from engine.energy import AnalyticMeter
from engine.online import run_stream
from utils.config import ROOT, load_yaml


def rolling_mase(err, steps, k):
    """Mean stream MASE over the k forecasts before and after each update step."""
    before = np.array([err[max(0, s - k + 1):s + 1].mean() for s in steps])
    after = np.array([err[s + 1:s + 1 + k].mean() if s + 1 < len(err) else np.nan for s in steps])
    return before, after


def check_exclusive(ecfg):
    if not ecfg.get("require_exclusive", True):
        return
    if any(c["type"] == "rapl" for c in ecfg["counters"]):
        have, node = len(os.sched_getaffinity(0)), os.cpu_count()
        if have < node:
            raise RuntimeError(f"RAPL measures the whole socket but this job holds {have} of {node} CPUs; "
                               "run on an exclusive node or set energy.require_exclusive=false")


def price_scale(measured_j, analytic_j):
    """Measured / analytic joules of one full-buffer update."""
    if measured_j <= 0 or analytic_j <= 0:
        raise RuntimeError(f"trial updates measured {measured_j:.3g} J each (analytic {analytic_j:.3g} J): "
                           "within noise of idle, so updates cannot be priced; is the node shared?")
    return measured_j / analytic_j


def scale_for(policy, measured_j, analytic_j):
    """The price scale, required only by policies that price updates (needs_cost). For the
    others an unmeasurable scale is recorded as None instead of failing the run."""
    try:
        return price_scale(measured_j, analytic_j)
    except RuntimeError:
        if policy.needs_cost:
            raise
        return None


def analytic_update_j(learner, cfg):
    """Analytic joules of one trial update on the current buffer; the hooks are removed afterwards."""
    meter, analytic = learner.meter, AnalyticMeter(learner.model, cfg)
    learner.meter = analytic
    try:
        return learner.trial_update()
    finally:
        learner.meter = meter
        analytic.remove()


def per_call(meter, fn, window_s, min_calls):
    """Incremental joules per call of fn, from calls repeated back to back in one window of at
    least window_s seconds (and min_calls calls). Returns {"j", "calls", "window_s"}."""
    calls = 0
    with meter.window("other") as i:
        t0 = time.perf_counter()
        while calls < min_calls or time.perf_counter() - t0 < window_s:
            fn()
            calls += 1
        window = time.perf_counter() - t0
    return {"j": meter.wait_settled(i) / calls, "calls": calls, "window_s": window}


def warm_up(learner, stream, n, ecfg, analytic_cfg):
    """First-call costs (allocator, CUDA context, caches) outside the measured loop, then the
    price scale on a full buffer. Returns a dict for the summary's energy block."""
    for t in range(min(len(stream), max(n, learner.buffer.maxlen))):
        ctx, y = stream[t]
        learner.predict(ctx)
        learner.push(ctx, y)
    learner.trial_update()  # cold: still pays first-call costs, so not part of any measurement
    learner.probe_grad()
    ctx, _ = stream[0]
    w, k = ecfg["scale_window_s"], ecfg["scale_min_updates"]
    calls = {"update": per_call(learner.meter, learner.trial_update, w, k),
             "probe": per_call(learner.meter, learner.probe_grad, w, k),
             "infer": per_call(learner.meter, lambda: learner.predict(ctx), w, k)}
    analytic = analytic_update_j(learner, analytic_cfg)
    learner.buffer.clear()
    u = calls["update"]
    return {"scale_measured_j": u["j"], "scale_analytic_j": analytic, "scale_updates": u["calls"],
            "scale_window_s": u["window_s"], "per_call": calls}


def main_cpus(first, sampler, threads):
    """Cores for the run: one per torch thread, from `first` upwards, skipping the sampler's.
    A multi-threaded run pinned to one core puts its whole thread team on that core."""
    cpus, c = [], first
    while len(cpus) < max(threads, 1):
        if c != sampler:
            cpus.append(c)
        c += 1
    return set(cpus)


def _total(d):
    return float(sum(d.values()))


def run_measured(cfg, learner, stream, policy):
    ecfg, meter = cfg["energy"], learner.meter
    sampler = meter.sampler
    cpus = os.sched_getaffinity(0)  # restored below: a sweep worker runs many runs in one process
    try:
        if cfg["loop"].get("budget"):
            raise ValueError("loop.budget is not supported with measured energy")
        check_exclusive(ecfg)
        pin = ecfg.get("pin") or {}
        if pin.get("main") is not None:
            os.sched_setaffinity(0, main_cpus(pin["main"], pin.get("sampler"), cfg.get("threads", 1)))
        a0 = time.perf_counter()
        time.sleep(ecfg["idle_s"])
        a1 = time.perf_counter()
        meter.p_idle = idle_power(sampler.snapshot(), [(a0, a1)])  # provisional, for online prices
        span0 = time.perf_counter()
        analytic_cfg = load_yaml(os.path.join(ROOT, "energy", ecfg["scale_reference"] + ".yaml"))
        with meter.window("other"):
            scale = warm_up(learner, stream, ecfg["warmup_steps"], ecfg, analytic_cfg)
        policy.price_scale = scale_for(policy, scale["scale_measured_j"], scale["scale_analytic_j"])
        summary, trace = run_stream(stream, learner, policy, cfg["loop"])
        span1 = time.perf_counter()
        time.sleep(ecfg["idle_s"])
        b1 = time.perf_counter()
        deadline = b1 + 5.0
        while not sampler.settled_after(b1):  # slow counters (NVML) must refresh past the window's end
            sampler.check()
            if time.perf_counter() > deadline:
                raise RuntimeError("energy counters did not refresh within 5 s after the idle window")
            time.sleep(0.005)
        sampler.check()
        samples = sampler.snapshot()
    finally:
        sampler.close()
        os.sched_setaffinity(0, cpus)

    pre, post = idle_power(samples, [(a0, a1)]), idle_power(samples, [(span1, b1)])
    p_idle = {d: 0.5 * (pre[d] + post[d]) for d in pre}
    drift = max(abs(pre[d] - post[d]) / max(abs(p_idle[d]), 1e-9) for d in pre)
    ev = meter.events()
    per, totals, closure = attribute(samples, ev, p_idle, (span0, span1))
    if closure > ecfg["max_closure"]:
        raise RuntimeError(f"energy closure error {closure:.3%} exceeds {ecfg['max_closure']:.1%}")

    top = ev["parent"] < 0
    incr = sum(per["incr"][d] for d in samples)
    system = {c: {"raw": _total(totals[c]["raw"]), "incr": _total(totals[c]["incr"])} for c in ORDER}

    def top_sum(kind):
        return float(incr[top & (ev["kind"] == kind)].sum())

    upd = top & (ev["kind"] == "update")
    update_j = top_sum("update") + top_sum("calib")
    summary.update(
        infer_j=system["inference"]["incr"],
        decide_j=system["decision"]["incr"],
        update_j=update_j,
        calib_j=top_sum("calib"),
        store_j=top_sum("store"),
        probe_j=float(incr[ev["kind"] == "probe"].sum()),
        other_j=system["other"]["incr"],
        adapt_j=system["adaptation"]["incr"] + system["decision"]["incr"],
        adapt_j_raw=system["adaptation"]["raw"] + system["decision"]["raw"],
        j_per_update=float(incr[upd].mean()) if upd.any() else 0.0,
        update_parts_j={"measured": update_j},
        energy={"p_idle_w": p_idle, "p_idle_pre_w": pre, "p_idle_post_w": post, "idle_drift": float(drift),
                "idle_drift_flag": bool(drift > ecfg["max_idle_drift"]), "closure": closure,
                "span_s": span1 - span0, "domains": list(samples), "system": system, "categories": totals,
                "price_scale": policy.price_scale, **scale},
    )

    steps = np.flatnonzero(trace["update"] > 0)
    before, after = rolling_mase(trace["err"], steps, cfg["loop"].get("rolling_k", 8))
    events = dict(ev)
    events.update({f"raw_{d}": per["raw"][d] for d in samples})
    events.update({f"incr_{d}": per["incr"][d] for d in samples})
    events.update(upd_step=steps, mase_before=before, mase_after=after,
                  buf_before=trace["buf_before"][steps], buf_after=trace["buf_after"][steps])
    sample_arrays = {}
    for d, (t, v) in samples.items():
        sample_arrays[f"{d}_t"], sample_arrays[f"{d}_j"] = t, v
    return summary, trace, {"events": events, "samples": sample_arrays}
