# TellCo User Analytics

User overview, engagement, experience and satisfaction analytics for TellCo's xDR data — an installable
Python package (`tellco`), a Streamlit dashboard, a feature store, MLflow model tracking and a Docker image.
Prepared by NextHikes IT Solutions.

## Quick start

```bash
pip install -e ".[dashboard]"                 # install the package and dashboard extras
tellco-run --data telcom_data.xlsx            # clean, cluster, score, train, track, build feature store
streamlit run app.py                # open the dashboard (reads telcom_data*.xlsx from the project root)
```

Export the scored table to MySQL at the same time:

```bash
tellco-run --data telcom_data.xlsx --mysql USER:PASSWORD@localhost:3306/tellco
```

## Package layout

| Module | Purpose |
|---|---|
| `tellco.pipeline` | Cleaning, per-customer aggregation, outlier treatment, k-means clustering, engagement / experience / satisfaction scores, robustness analyses |
| `tellco.modeling` | Regression models, loss convergence, permutation importance, SHAP, model save/load |
| `tellco.tracking` | MLflow run logging (parameters, metrics, loss steps, artifacts) |
| `tellco.feature_store` | Versioned parquet snapshots with a JSON registry |
| `tellco.db` | Export to MySQL (or any SQLAlchemy database) with a verification `SELECT` |
| `tellco.cli` | `tellco-run` command |


## Notes on the data

* `Dur. (ms)` is in **seconds** (it matches End − Start); `Dur. (ms).1` is the true millisecond value and is dropped.
* Rows without `MSISDN/Number` cannot be attributed to a customer and are dropped.
* `Total DL (Bytes)` excludes `Other DL`; `Total UL (Bytes)` includes `Other UL`.
* The satisfaction score is a behavioural proxy (distance-based), not measured satisfaction.
