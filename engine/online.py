"""Generic online loop: forecast, reveal target, store, let the policy decide, update.

Energy accounting per step:
  inference : forecast forward pass (identical for every policy)
  storage   : writing the newest window (or cached prefix) into the buffer
  update    : committed adapter updates, plus a one-off cost calibration for
              policies that price updates or when a budget is active
  probe     : energy a policy spends computing its own decision signal
  decide    : every policy's should_update runs inside a `decide` window (recorded by
              sampled meters; the analytic meter records nothing)
The update price (cost_est) is updated only from settled energies: a sampled meter's
value is final once its counters refresh past the update, which on NVML takes up to
~100 ms. The analytic meter settles immediately, so its runs are unchanged.
The per-step error comes from the stream (stream.error): MASE for forecasting, the
misclassification rate for vision.
An optional energy budget (token bucket with per-step income, e.g. harvested energy)
blocks any update the bucket cannot pay for, for every policy alike. It is either absolute
({capacity_j, income_j}; the bucket starts full) or relative to the calibrated update cost
({income_frac, capacity_updates}, in units of the update-price estimate, calibrated once the
buffer is full; the bucket starts empty at calibration, so every policy
receives the same energy whatever the capacity). Probes are paid from the bucket too.
"""
import numpy as np


def run_stream(stream, learner, policy, cfg, on_step=None):
    budget = cfg.get("budget") or {}
    relative = "income_frac" in budget or "capacity_updates" in budget
    if relative and ("income_j" in budget or "capacity_j" in budget):
        raise ValueError(f"budget is either absolute (joules) or relative (update costs), got {budget}")
    bucket = -np.inf if relative else budget.get("capacity_j", np.inf)  # relative: nothing before calibration
    income = 0.0 if relative else budget.get("income_j", 0.0)
    capacity = np.inf if relative else budget.get("capacity_j", np.inf)
    meter = learner.meter
    buffer_eval = cfg.get("buffer_eval", False)
    cost_est, alpha_cost = None, cfg["cost_ema"]
    tr = {k: [] for k in ("err", "update", "update_j", "infer_j", "store_j", "probe_j", "update_s", "infer_s",
                          "bucket", "buf_before", "buf_after")}
    calib_j = 0.0
    probe = [0.0]  # energy policies spend on their own decision signals, charged per step
    parts_sum = {}
    pending, lags = [], []  # (event id, step) of updates whose energy has not settled yet

    for t in range(len(stream)):
        meter.step = t
        ctx, y = stream[t]
        pred, e_inf, s_inf = learner.predict(ctx)
        err = stream.error(pred, y)
        e_store = learner.push(ctx, y) if policy.adapts else 0.0
        policy.observe(t, err)

        calib_at = learner.buffer.maxlen if relative else cfg["min_buffer"]  # relative: price a full-buffer update
        if cost_est is None and (policy.needs_cost or budget) and len(learner.buffer) >= calib_at:
            learner.trial_update()
            cost_est = calib_j = meter.wait_settled(meter.last_id)
            bucket = 0.0 if relative else bucket

        if relative and cost_est is not None:
            income, capacity = budget["income_frac"] * cost_est, budget["capacity_updates"] * cost_est
        bucket = min(capacity, bucket + income)
        did, e_upd, s_upd = False, 0.0, 0.0
        b_before = b_after = np.nan
        ready = len(learner.buffer) >= cfg["min_buffer"]
        probe[0] = 0.0
        view = {"cost_est": cost_est, "learner": learner,
                "charge": lambda j: probe.__setitem__(0, probe[0] + j),
                "peek": (lambda t=t: stream[t + 1] if t + 1 < len(stream) else None)}
        if hasattr(stream, "scale"):
            view["scale"] = stream.scale  # forecasting only (the oracle scores its preview with MASE)
        if ready and (bucket >= cost_est if cost_est is not None else not relative):
            with meter.window("decide"):
                want = policy.should_update(t, view)
            if want:
                if buffer_eval:
                    with meter.window("other"):
                        b_before = learner.buffer_loss()
                e_upd, s_upd, parts = learner.update()
                pending.append((meter.last_id, t))
                if buffer_eval:
                    with meter.window("other"):
                        b_after = learner.buffer_loss()
                policy.after_update(t, e_upd)
                bucket -= e_upd
                for k, v in parts.items():
                    parts_sum[k] = parts_sum.get(k, 0.0) + v
                did = True
        for item in list(pending):
            v = meter.settled(item[0])
            if v is not None:
                cost_est = v if cost_est is None else (1 - alpha_cost) * cost_est + alpha_cost * v
                lags.append(t - item[1])
                pending.remove(item)

        bucket -= probe[0]
        for k, v in zip(tr, (err, did, e_upd, e_inf, e_store, probe[0], s_upd, s_inf, bucket, b_before, b_after)):
            tr[k].append(v)
        if on_step is not None:
            on_step(t)  # outside every meter window, so nothing it does is charged

    tr = {k: np.asarray(v, dtype=np.float64) for k, v in tr.items()}
    n_upd = int(tr["update"].sum())
    summary = {
        "error": float(tr["err"].mean()),
        "n_updates": n_upd,
        "update_rate": n_upd / len(stream),
        "update_j": float(tr["update_j"].sum() + calib_j),
        "store_j": float(tr["store_j"].sum()),
        "probe_j": float(tr["probe_j"].sum()),
        "infer_j": float(tr["infer_j"].sum()),
        "adapt_j": float(tr["update_j"].sum() + calib_j + tr["store_j"].sum() + tr["probe_j"].sum()),
        "j_per_update": float(tr["update_j"].sum() / max(n_upd, 1)),
        "s_per_update": float(tr["update_s"].sum() / max(n_upd, 1)),
        "update_s": float(tr["update_s"].sum()),
        "calib_j": calib_j,
        "update_parts_j": parts_sum,
        "realizable": policy.realizable,
        "price_lag_mean": float(np.mean(lags)) if lags else 0.0,
    }
    if hasattr(stream, "scale"):
        summary["mase"] = summary["error"]  # forecasting streams: the error is MASE
    return summary, tr
