from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import mlflow

def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(8192)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def start_run(experiment: str, tracking_uri: str, run_name: str, tags: Dict[str, Any] | None = None):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    return mlflow.start_run(run_name=run_name, tags=tags)

def log_dict(name: str, payload: Dict[str, Any]) -> None:
    mlflow.log_text(json.dumps(payload, indent=2), f"{name}.json")
