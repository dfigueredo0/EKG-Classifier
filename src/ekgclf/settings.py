from __future__ import annotations

import json
import logging.config
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator
from omegaconf import OmegaConf

class DataPaths(BaseModel):
    raw_ptbxl: str
    raw_mitbih: str
    processed: str
    splits_dir: str

class Signals(BaseModel):
    resample_hz: int = 500
    bandpass: dict = Field(default_factory=lambda: {"low": 0.5, "high": 40.0, "order": 4})
    zscore: bool = True
    required_leads: list[str] = Field(default_factory=list)

    @field_validator("resample_hz")
    @classmethod
    def _resample_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("resample_hz must be > 0")
        return v

class Split(BaseModel):
    seed: int = 1337
    patient_level: bool = True
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    @field_validator("train", "val", "test")
    @classmethod
    def _prob(cls, v: float) -> float:
        if v <= 0 or v >= 1:
            raise ValueError("split fractions must be in (0,1)")
        return v

    @field_validator("test")
    @classmethod
    def _sum_to_one(cls, v: float, info):
        # Cannot access other fields here; validate later in load()
        return v

class DataConfig(BaseModel):
    paths: DataPaths
    signals: Signals
    split: Split
    augment: dict
    ptbxl: dict
    mitbih: dict
    io: dict

class ModelConfig(BaseModel):
    model: dict
    head: dict

class TrainConfig(BaseModel):
    trainer: dict
    loss: dict
    class_weights: dict
    selective: dict
    mlflow: dict

class EvalConfig(BaseModel):
    eval: dict
    outputs: dict

def load_yaml(path: str | Path) -> dict:
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    return OmegaConf.to_container(cfg, resolve=True)

def setup_logging(config_path: str | Path) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logging.config.dictConfig(cfg)

def validate_splits(cfg: DataConfig) -> None:
    s = cfg.split
    total = s.train + s.val + s.test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0 (got {total})")

def dump_runtime_config(out_dir: Path, *cfgs) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in cfgs:
        p = out_dir / f"runtime_{c.__class__.__name__.lower()}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(c.model_dump(), f, indent=2)