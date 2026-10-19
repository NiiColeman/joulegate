"""Uncertainty-based performance forecasting (Regol et al., When to retrain a machine learning model,
ICML 2025), adapted to online adapter updates and priced in energy.

Every step gives a_{i,j}: the error at step j of the adapter last updated at step i (i = 0: pretrained).
A linear model mu(r) of a_{i,j} on r = [1, i, j, j - i, z_{i,j}] (steps divided by SCALE; z_{i,j} the
input shift of step j against step i, the paper's z_shift: mean over channels of |mu_j - mu_i| / sd_i)
is refit every `refit` steps by ridge least squares on every pair seen. The paper fits ElasticNetCV;
scikit-learn cannot be imported in this environment (joblib missing), and a fixed small ridge penalty
keeps the collinear [i, j, j - i] identifiable in the same way.

The paper compares delta-quantiles of the total cost of keeping and of retraining. With its constant-
variance Gaussian, both branches over the next `horizon` steps have the same variance, so the
comparison reduces to the means:
    retrain iff  sum_{k < H} [mu(r, t + k, z_now) - mu(t, t + k, 0)]  >  lam * C
with C the update joules (Priced). The paper's offline history of trained models becomes a warm-up:
until `warmup_models` adapters exist, update every `warmup_period` steps (charged like any update).
"""
import numpy as np

from methods.base import Priced

SCALE = 100.0  # steps per feature unit


class UPF(Priced):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.X, self.y = [], []
        self.r, self.n_models, self.last_err = 0, 0, None
        self.ref = self.cur = None   # input statistics of the current adapter's step / of this step
        self.w, self.fit_at = None, None

    @staticmethod
    def features(i, j, z):
        j = np.atleast_1d(np.asarray(j, dtype=float))
        i = np.broadcast_to(float(i), j.shape)
        z = np.broadcast_to(float(z), j.shape)
        return np.stack([np.ones_like(j), i / SCALE, j / SCALE, (j - i) / SCALE, z], -1)

    def observe(self, t, err):
        self.last_err = err

    def _z(self, learner):
        mu = learner.last_mu.reshape(-1).double().cpu().numpy()
        sd = learner.last_sd.reshape(-1).double().cpu().numpy()
        self.cur = (mu, sd)
        if self.ref is None:
            self.ref = self.cur
        m0, s0 = self.ref
        return float(np.mean(np.abs(mu - m0) / np.maximum(s0, 1e-12)))

    def _fit(self):
        X, y = np.concatenate(self.X), np.asarray(self.y)
        A = X.T @ X + self.cfg["ridge"] * np.eye(X.shape[1])
        self.w = np.linalg.solve(A, X.T @ y)

    def should_update(self, t, ctx):
        z = self._z(ctx["learner"])
        self.X.append(self.features(self.r, t, z))
        self.y.append(self.last_err)
        if self.n_models < self.cfg["warmup_models"]:
            return t - self.r >= self.cfg["warmup_period"]
        price = self.price(ctx["cost_est"])
        if price is None:
            return False
        if self.w is None or t - self.fit_at >= self.cfg["refit"]:
            self._fit()
            self.fit_at = t
        k = t + np.arange(self.cfg["horizon"])
        keep = self.features(self.r, k, z) @ self.w
        retrain = self.features(t, k, 0.0) @ self.w
        return float((keep - retrain).sum()) > self.cfg["lam"] * price

    def after_update(self, t, joules):
        self.r, self.n_models = t, self.n_models + 1
        self.ref = self.cur          # the new adapter was trained on data up to this step
