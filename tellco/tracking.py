"""Experiment tracking (Task 4.7): MLflow runs plus a self-contained JSON run log."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd


def log_run(run_log: dict, loss: pd.DataFrame, results: pd.DataFrame, model_path: str | Path,
            out_dir: str | Path = "outputs", experiment: str = "tellco-satisfaction") -> str:
    """Record one training run in MLflow (params, metrics, loss steps, artifacts) and as ``run_log.json``."""
    import mlflow

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_log.json").write_text(json.dumps(run_log, indent=2, default=str))
    loss.to_csv(out / "loss_curve.csv", index=False)
    results.to_csv(out / "model_comparison.csv")

    mlflow.set_tracking_uri(f"sqlite:///{(out / 'mlflow.db').resolve()}")
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(out / "mlartifacts").resolve().as_uri())
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_log["run_id"]) as run:
        mlflow.set_tags({"code_version": run_log["code_version"], "target": run_log["source"]["target"],
                         "data_fingerprint": run_log["source"]["data_fingerprint"],
                         "model": run_log["model"], "started": run_log["start_time"], "ended": run_log["end_time"]})
        mlflow.log_params({**run_log["parameters"], "split": run_log["split"],
                           "n_customers": run_log["source"]["n_customers"]})
        mlflow.log_metrics({k: float(v) for k, v in run_log["metrics"].items()})
        from mlflow.entities import Metric
        from mlflow.tracking import MlflowClient
        ts = int(time.time() * 1000)
        batch = [Metric(name, float(r[name]), ts, int(r["iteration"]))
                 for _, r in loss.iterrows() for name in ("train_loss", "test_loss")]
        client = MlflowClient()
        for i in range(0, len(batch), 1000):           # the API accepts at most 1000 metrics per call
            client.log_batch(run.info.run_id, metrics=batch[i:i + 1000])
        for f in [out / "run_log.json", out / "loss_curve.csv", out / "model_comparison.csv", Path(model_path)]:
            if f.exists():
                mlflow.log_artifact(str(f))
        return run.info.run_id


def list_runs(out_dir: str | Path = "outputs", experiment: str = "tellco-satisfaction") -> pd.DataFrame:
    """All tracked runs (newest first); empty frame if tracking has not been used yet."""
    db = Path(out_dir) / "mlflow.db"
    if not db.exists():
        return pd.DataFrame()
    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{db.resolve()}")
    if mlflow.get_experiment_by_name(experiment) is None:
        return pd.DataFrame()
    runs = mlflow.search_runs(experiment_names=[experiment], order_by=["start_time DESC"])
    keep = [c for c in runs.columns if c.startswith(("metrics.", "params.", "tags.")) or c in
            ("run_id", "start_time", "end_time", "status")]
    return runs[keep]


def load_run_log(out_dir: str | Path = "outputs") -> dict | None:
    p = Path(out_dir) / "run_log.json"
    return json.loads(p.read_text()) if p.exists() else None
