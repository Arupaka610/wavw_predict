"""
Configuration loader using PyYAML with a simple namespace wrapper.
Provides dict-like access via attribute notation (cfg.training.lr, etc.).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class DictConfig:
    """Lightweight namespace that supports dot-access and dict merging."""

    def __init__(self, data: dict):
        for k, v in data.items():
            if isinstance(v, dict):
                setattr(self, k, DictConfig(v))
            elif isinstance(v, list):
                setattr(self, k, [DictConfig(i) if isinstance(i, dict) else i for i in v])
            else:
                setattr(self, k, v)

    def __repr__(self):
        return f"DictConfig({self.__dict__})"

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: str | Path) -> DictConfig:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}
    return DictConfig(raw)


def merge_configs(*configs: DictConfig) -> DictConfig:
    merged: dict = {}
    for cfg in configs:
        merged = _deep_merge(merged, cfg.__dict__ if isinstance(cfg, DictConfig)
                             else cfg)
    return DictConfig(merged)


def load_base_with_override(override_path: str | Path) -> DictConfig:
    base_path = Path(__file__).parent.parent.parent / "configs" / "base.yaml"
    base = load_config(base_path)
    override = load_config(override_path)
    return merge_configs(base, override)


def config_to_dict(cfg: DictConfig) -> dict:
    def _unwrap(v):
        if isinstance(v, DictConfig):
            return {k2: _unwrap(v2) for k2, v2 in v.__dict__.items()}
        if isinstance(v, list):
            return [_unwrap(i) for i in v]
        return v
    return {k: _unwrap(v) for k, v in cfg.__dict__.items()}
