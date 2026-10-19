"""Policy registry. `policy:` in configs/methods/<name>.yaml selects the class."""
from methods.cara import CARA
from methods.d3a import D3A
from methods.drift import Drift
from methods.frozen import Frozen
from methods.oracle import Oracle
from methods.periodic import Periodic
from methods.upf import UPF
from methods.vou import VoU
from methods.vou_err import VoUErr
from methods.vou_gated import VoUGated
from methods.vou_resid import VoUResid

POLICIES = {
    "frozen": Frozen,
    "cara": CARA,
    "upf": UPF,
    "d3a": D3A,
    "periodic": Periodic,
    "drift": Drift,
    "vou": VoU,
    "vou_err": VoUErr,
    "oracle": Oracle,
    "vou_resid": VoUResid,
    "vou_gated": VoUGated,
}


def build_policy(cfg):
    return POLICIES[cfg["policy"]](cfg)
