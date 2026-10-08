"""Regression models for the satisfaction proxy (Task 4.3), with SHAP and loss convergence."""
from __future__ import annotations

import hashlib
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .pipeline import MODEL_FEATURES, PIPELINE_VERSION, RANDOM_STATE

GB_PARAMS = {"n_estimators": 200, "max_depth": 3, "random_state": RANDOM_STATE}


def code_version(fallback_key: str = "") -> str:
    """Short git commit (``-dirty`` if uncommitted changes) or a data fingerprint when git is unavailable."""
    try:
        v = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        if subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip():
            v += "-dirty"
        return v
    except Exception:  # noqa: BLE001
        return "no-git|data:" + hashlib.sha256(fallback_key.encode()).hexdigest()[:12]


def train_models(comb: pd.DataFrame, data_key: str = "", cv_rows: int = 20_000, rf_trees: int = 100,
                 shap_rows: int = 2_000) -> dict:
    """Fit and compare six regressors on an 80/20 split; return metrics, loss curve, importances, SHAP values."""
    started = datetime.now()
    X, y = comb[MODEL_FEATURES], comb["Satisfaction Score"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    cv_idx = np.random.RandomState(RANDOM_STATE).choice(len(Xtr), size=min(cv_rows, len(Xtr)), replace=False)
    models = {
        "Dummy (mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso": Lasso(alpha=0.001, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=rf_trees, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(**GB_PARAMS),
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted = [], {}
    for name, m in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", m)])
        t0 = time.time()
        pipe.fit(Xtr, ytr)
        fit_s = time.time() - t0
        pte, ptr = pipe.predict(Xte), pipe.predict(Xtr)
        cv = cross_val_score(Pipeline([("scaler", StandardScaler()), ("model", m.__class__(**m.get_params()))]),
                             Xtr.iloc[cv_idx], ytr.iloc[cv_idx], cv=kf, scoring="r2", n_jobs=1)
        rows.append({"Model": name, "MAE": mean_absolute_error(yte, pte),
                     "RMSE": float(np.sqrt(mean_squared_error(yte, pte))),
                     "R² (test)": r2_score(yte, pte), "R² (train)": r2_score(ytr, ptr),
                     "Train–test gap": r2_score(ytr, ptr) - r2_score(yte, pte),
                     "CV R² mean": cv.mean(), "CV R² std": cv.std(), "Fit time (s)": fit_s})
        fitted[name] = pipe
    results = pd.DataFrame(rows).set_index("Model")

    gb = fitted["Gradient Boosting"]
    scaler, gbm = gb.named_steps["scaler"], gb.named_steps["model"]
    Ztr, Zte = scaler.transform(Xtr), scaler.transform(Xte)
    loss = pd.DataFrame({
        "iteration": np.arange(1, gbm.n_estimators + 1),
        "train_loss": [0.5 * mean_squared_error(ytr, p) for p in gbm.staged_predict(Ztr)],
        "test_loss": [0.5 * mean_squared_error(yte, p) for p in gbm.staged_predict(Zte)],
    })
    within_1pct = int(np.argmax(loss["test_loss"].values <= loss["test_loss"].iloc[-1] * 1.01)) + 1

    samp = Xte.sample(n=min(5000, len(Xte)), random_state=RANDOM_STATE)
    perm = permutation_importance(gb, samp, yte.loc[samp.index], n_repeats=3, random_state=RANDOM_STATE, n_jobs=1)
    importance = pd.DataFrame({"Permutation": pd.Series(perm.importances_mean, index=MODEL_FEATURES),
                               "Native (impurity)": pd.Series(gbm.feature_importances_, index=MODEL_FEATURES)})

    import shap  # imported lazily: optional heavy dependency
    sx = Xte.sample(n=min(shap_rows, len(Xte)), random_state=RANDOM_STATE)
    expl = shap.TreeExplainer(gbm)
    sv = expl.shap_values(scaler.transform(sx))
    recon_err = float(np.abs(sv.sum(axis=1) + np.ravel(expl.expected_value)[0] - gbm.predict(scaler.transform(sx))).max())
    shap_info = {"values": pd.DataFrame(sv, columns=MODEL_FEATURES, index=sx.index), "data": sx,
                 "mean_abs": pd.Series(np.abs(sv).mean(axis=0), index=MODEL_FEATURES).sort_values(ascending=False),
                 "additivity_error": recon_err}

    ended = datetime.now()
    best = results.drop(index="Dummy (mean)")["R² (test)"].idxmax()
    gbr = results.loc["Gradient Boosting"]
    run_log = {
        "run_id": f"satisfaction_model_{started.strftime('%Y%m%d_%H%M%S')}",
        "code_version": code_version(data_key), "pipeline_version": PIPELINE_VERSION,
        "start_time": started.isoformat(timespec="seconds"), "end_time": ended.isoformat(timespec="seconds"),
        "duration_seconds": round((ended - started).total_seconds(), 1),
        "source": {"data_fingerprint": hashlib.sha1(data_key.encode()).hexdigest()[:12],
                   "n_customers": len(comb), "features": MODEL_FEATURES,
                   "target": "Satisfaction Score = mean(engagement score, experience score)"},
        "model": "Gradient Boosting", "parameters": GB_PARAMS, "split": "80/20, random_state=42",
        "metrics": {"MAE": round(float(gbr["MAE"]), 6), "RMSE": round(float(gbr["RMSE"]), 6),
                    "R2": round(float(gbr["R² (test)"]), 6),
                    "final_train_loss": round(float(loss["train_loss"].iloc[-1]), 8),
                    "final_test_loss": round(float(loss["test_loss"].iloc[-1]), 8),
                    "iterations_to_within_1pct_of_final_test_loss": within_1pct},
        "best_by_test_r2": best,
    }
    return {"results": results, "loss": loss, "importance": importance, "shap": shap_info,
            "run_log": run_log, "model": gb, "n_test": len(Xte)}


def predict_satisfaction(model, features: dict | pd.DataFrame) -> np.ndarray:
    """Predict the satisfaction proxy for one or many customers with a trained pipeline."""
    frame = pd.DataFrame([features]) if isinstance(features, dict) else features
    return model.predict(frame[MODEL_FEATURES])


import json
import pickle  # noqa: F401  (kept for clarity of what can fail)


class IncompatibleModelError(RuntimeError):
    """A saved model cannot be loaded here (usually a scikit-learn version change)."""


def save_model(model, path: str | Path) -> Path:
    import joblib
    import sklearn

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    # sidecar so a version mismatch is detected before unpickling
    path.with_suffix(".meta.json").write_text(json.dumps({"sklearn": sklearn.__version__}))
    return path


def load_model(path: str | Path):
    import joblib
    import sklearn

    path = Path(path)
    meta = path.with_suffix(".meta.json")
    if meta.exists():
        saved = json.loads(meta.read_text()).get("sklearn")
        if saved and saved != sklearn.__version__:
            raise IncompatibleModelError(
                f"{path.name} was saved with scikit-learn {saved}, but this environment has "
                f"{sklearn.__version__}. Re-run `tellco-run --data <file>` to rebuild it."
            )
    try:
        return joblib.load(path)
    except Exception as exc:  # old pickles fail with ModuleNotFoundError, AttributeError, etc.
        raise IncompatibleModelError(
            f"{path.name} could not be loaded with scikit-learn {sklearn.__version__} ({type(exc).__name__}: {exc}). "
            f"Re-run `tellco-run --data <file>` to rebuild it."
        ) from exc
