"""Update-policy interface. One file per policy; the online loop only calls these hooks.

observe(t, err)            called every step once the forecast error is known
should_update(t, ctx)      decide whether to spend energy on an adapter update now
after_update(t, joules)    called after a committed update with its measured energy

`ctx` carries: cost_est (EMA of measured update joules, or None), learner, peek().
"""


class Policy:
    adapts = True       # False only for the frozen baseline (no buffer is kept)
    needs_cost = False  # True if the policy reads ctx["cost_est"]
    realizable = True   # False for oracles that look at future targets

    def __init__(self, cfg):
        self.cfg = cfg

    def observe(self, t, err):
        pass

    def should_update(self, t, ctx):
        raise NotImplementedError

    def after_update(self, t, joules):
        pass


class Priced(Policy):
    """A policy that prices updates in joules. Measured runs set price_scale = measured / analytic
    joules of the same update (engine/measured.py), so lam keeps its analytic meaning."""
    needs_cost = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self.price_scale = None

    def price(self, cost_est):
        if cost_est is None:
            return None
        return cost_est if self.price_scale is None else cost_est / self.price_scale
