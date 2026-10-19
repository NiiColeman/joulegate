"""YAML config loading with composable groups and dotted CLI overrides.

A run config is base.yaml plus one file from each group (dataset, method, energy),
plus `--set a.b=value` overrides. Values in overrides are parsed as YAML.
"""
import copy
import os

import yaml

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
GROUPS = {"dataset": "datasets", "method": "methods", "energy": "energy"}  # group -> folder


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg, key, value):
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def build(dataset=None, method=None, energy=None, overrides=(), backbone=None):
    base = "base.yaml"
    if dataset is not None:
        if load_yaml(os.path.join(ROOT, "datasets", dataset + ".yaml")).get("task") == "vision":
            base = "base_vision.yaml"
    cfg = load_yaml(os.path.join(ROOT, base))
    if backbone not in (None, "patchmixer"):  # a released forecaster: configs/base_<backbone>.yaml over base.yaml
        cfg = merge(cfg, load_yaml(os.path.join(ROOT, f"base_{backbone}.yaml")))
    cfg.setdefault("backbone", "patchmixer")
    chosen = {"dataset": dataset, "method": method, "energy": energy}
    for group, folder in GROUPS.items():
        name = chosen[group] or cfg.get(group, {}).get("name")
        group_cfg = load_yaml(os.path.join(ROOT, folder, name + ".yaml"))
        cfg[group] = merge(cfg.get(group, {}), group_cfg)
        cfg[group]["name"] = name
    dov = cfg.pop("dataset_overrides", {})  # a backbone's stream settings, all datasets then this one
    for key, value in {**dov.get("all", {}), **dov.get(cfg["dataset"]["name"], {})}.items():
        set_dotted(cfg["dataset"], key, value)
    for key, value in cfg["dataset"].get("overrides", {}).items():  # e.g. a dataset's own checkpoint
        set_dotted(cfg, key, value)
    for item in overrides:
        key, raw = item.split("=", 1)
        set_dotted(cfg, key, yaml.safe_load(raw))
    return cfg
