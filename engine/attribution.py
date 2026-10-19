"""Turn sampled cumulative energy counters and event windows into joules.

A domain's samples are change points (t_k, E_k): the time a cumulative counter was first
seen at a new value, and that value. Energy between two times is the counter linearly
interpolated at both ends. Nothing here reads hardware, so it is tested with synthetic
counters.

Only top-level events (parent < 0) are totalled by category; nested events such as
VoU's probe inside a decide window are kept for diagnostics. Time between top-level
windows inside the accounting span counts as `other`.
"""
import numpy as np

CATEGORIES = {"infer": "inference", "decide": "decision", "update": "adaptation",
              "store": "adaptation", "calib": "adaptation", "other": "other"}
ORDER = ("inference", "decision", "adaptation", "other")


def counter_at(t, v, when):
    """Cumulative joules at time(s) `when`: linear between change points, flat outside."""
    return np.interp(when, t, v)


def energy_between(samples, t0, t1):
    """{domain: joules} consumed between t0 and t1 (scalars or equal-length arrays)."""
    return {d: counter_at(t, v, t1) - counter_at(t, v, t0) for d, (t, v) in samples.items()}


def idle_power(samples, windows):
    """{domain: watts}: mean power over the idle windows [(t0, t1), ...]."""
    total = sum(b - a for a, b in windows)
    out = {}
    for d in samples:
        joules = sum(float(energy_between({d: samples[d]}, a, b)[d]) for a, b in windows)
        out[d] = joules / total
    return out


def attribute(samples, events, p_idle, span):
    """Per-event and per-category energy. Returns (per_event, totals, closure).

    per_event: {"raw": {domain: array}, "incr": {domain: array}}, one value per event.
    totals:    {category: {"raw": {domain: J}, "incr": {domain: J}}}.
    closure:   |sum of categories - counter energy over span| / |counter energy over span|, raw.
    """
    t0, t1, kind = events["t0"], events["t1"], events["kind"]
    dur = t1 - t0
    raw = energy_between(samples, t0, t1)
    incr = {d: raw[d] - p_idle[d] * dur for d in samples}

    top = events["parent"] < 0
    unknown = sorted(set(kind[top]) - set(CATEGORIES))
    if unknown:
        raise ValueError(f"unknown top-level event kind(s): {unknown}")
    order = np.argsort(t0[top])
    s, e = t0[top][order], t1[top][order]
    if len(s) and np.any(s[1:] < e[:-1] - 1e-9):
        raise ValueError("top-level event windows overlap")
    if len(s) and (s[0] < span[0] - 1e-9 or e[-1] > span[1] + 1e-9):
        raise ValueError("an event window lies outside the accounting span")

    gap0, gap1 = np.r_[span[0], e], np.r_[s, span[1]]
    gap_raw = energy_between(samples, gap0, gap1)
    totals = {c: {"raw": {}, "incr": {}} for c in ORDER}
    for d in samples:
        for c in ORDER:
            totals[c]["raw"][d] = totals[c]["incr"][d] = 0.0
        for k, c in CATEGORIES.items():
            m = top & (kind == k)
            totals[c]["raw"][d] += float(raw[d][m].sum())
            totals[c]["incr"][d] += float(incr[d][m].sum())
        totals["other"]["raw"][d] += float(gap_raw[d].sum())
        totals["other"]["incr"][d] += float((gap_raw[d] - p_idle[d] * (gap1 - gap0)).sum())

    span_raw = sum(float(v) for v in energy_between(samples, span[0], span[1]).values())
    got = sum(totals[c]["raw"][d] for c in ORDER for d in samples)
    closure = abs(got - span_raw) / max(abs(span_raw), 1e-12)
    return {"raw": raw, "incr": incr}, totals, float(closure)
