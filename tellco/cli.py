"""Command line entry point: ``tellco-run --data telcom_data.xlsx --out outputs``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .db import export_scores, mysql_url, scores_frame
from .feature_store import save_features
from .modeling import save_model, train_models
from .pipeline import read_raw, run_pipeline
from .tracking import log_run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tellco-run", description="Run the TellCo analytics pipeline end to end.")
    ap.add_argument("--data", required=True, help="Path to the xDR extract (.xlsx or .csv)")
    ap.add_argument("--out", default="outputs", help="Output directory (default: outputs)")
    ap.add_argument("--feature-store", default="feature_store", help="Feature store directory")
    ap.add_argument("--no-train", action="store_true", help="Skip model training and tracking")
    ap.add_argument("--mysql", metavar="USER:PASSWORD@HOST:PORT/DB", help="Also export the scored table to MySQL")
    ap.add_argument("--version", action="version", version=f"tellco {__version__}")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    P = run_pipeline(read_raw(a.data))
    comb = P["comb"]
    frame = scores_frame(comb)
    frame.to_csv(out / "customer_scores.csv", index=False)
    entry = save_features(comb, a.feature_store, source=str(a.data))
    summary = {"customers": len(comb), "sessions": int(P["q"]["n_sessions"]), "feature_store_version": entry["version"]}

    if not a.no_train:
        M = train_models(comb, data_key=str(a.data))
        model_path = save_model(M["model"], out / "models" / f"{M['run_log']['run_id']}.joblib")
        M["run_log"]["model_path"] = str(model_path)
        summary["mlflow_run_id"] = log_run(M["run_log"], M["loss"], M["results"], model_path, out)
        summary["test_r2"] = M["run_log"]["metrics"]["R2"]

    if a.mysql:
        cred, _, rest = a.mysql.rpartition("@")
        user, _, pw = cred.partition(":")
        hostport, _, db = rest.partition("/")
        host, _, port = hostport.partition(":")
        info = export_scores(frame, mysql_url(user, pw, host or "localhost", port or 3306, db),
                             log_path=out / "mysql_export.json")
        summary["mysql_rows"] = info["rows_in_table"]

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
