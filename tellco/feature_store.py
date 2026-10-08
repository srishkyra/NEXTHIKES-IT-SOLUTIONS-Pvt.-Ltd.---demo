"""A small, reusable feature store: versioned parquet snapshots plus a JSON registry."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .pipeline import PIPELINE_VERSION

FEATURES = [
    "xDR Sessions", "Total Duration (s)", "Total Data Volume (Bytes)",
    "Avg TCP Retransmission (Bytes)", "Avg RTT (ms)", "Avg Throughput (kbps)",
    "Most Frequent Handset", "Engagement Cluster", "Engagement Score",
    "Experience Cluster", "Experience Score", "Satisfaction Cluster", "Satisfaction Score", "Satisfaction Score (scaled)",
]


def save_features(comb: pd.DataFrame, root: str | Path = "feature_store", features: list[str] | None = None,
                  source: str = "") -> dict:
    """Write customer features (MSISDN as string key) and append an entry to ``registry.json``."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cols = features or FEATURES
    fs = comb[cols].copy()
    fs.index = fs.index.map(lambda x: str(int(x)))
    fs.index.name = "MSISDN"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = root / f"customer_features_v{stamp}.parquet"
    fs.to_parquet(path)
    entry = {
        "version": stamp, "pipeline_version": PIPELINE_VERSION, "path": str(path), "rows": len(fs),
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": source, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
        "features": {c: str(fs[c].dtype) for c in fs.columns},
    }
    reg_path = root / "registry.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else []
    reg.append(entry)
    reg_path.write_text(json.dumps(reg, indent=2))
    return entry


def load_registry(root: str | Path = "feature_store") -> list[dict]:
    p = Path(root) / "registry.json"
    return json.loads(p.read_text()) if p.exists() else []


def load_features(root: str | Path = "feature_store", version: str | None = None,
                  columns: list[str] | None = None) -> pd.DataFrame:
    """Load a stored snapshot (latest by default), optionally selecting columns."""
    reg = load_registry(root)
    if not reg:
        raise FileNotFoundError(f"No feature-store registry in {root}")
    entry = reg[-1] if version is None else next(e for e in reg if e["version"] == version)
    return pd.read_parquet(entry["path"], columns=columns)
