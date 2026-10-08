"""Database export of the final scored table (Task 4.6)."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

EXPORT_COLS = ["Engagement Score", "Experience Score", "Satisfaction Score",
               "Engagement Tier", "Experience Group", "Satisfaction Group"]


def scores_frame(comb: pd.DataFrame) -> pd.DataFrame:
    """The table to export: customer id plus engagement, experience and satisfaction scores."""
    f = comb[EXPORT_COLS].reset_index()
    f["MSISDN/Number"] = f["MSISDN/Number"].astype("int64")
    return f


def mysql_url(user: str, password: str, host: str = "localhost", port: int | str = 3306, db: str = "tellco") -> str:
    from urllib.parse import quote_plus
    return f"mysql+pymysql://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{db}"


def export_scores(frame: pd.DataFrame, url: str, table: str = "user_scores", log_path: str | Path | None = None,
                  preview_rows: int = 10) -> dict:
    """Write ``frame`` to ``table`` (replace) and return a verification block with a real SELECT result."""
    from sqlalchemy import create_engine, text
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
        raise ValueError("Table name must be letters, digits and underscores only")
    engine = create_engine(url)
    frame.to_sql(table, engine, if_exists="replace", index=False, chunksize=5000, method="multi")
    query = f"SELECT * FROM `{table}` ORDER BY `Satisfaction Score` DESC LIMIT {int(preview_rows)}" \
        if engine.dialect.name == "mysql" else \
        f'SELECT * FROM "{table}" ORDER BY "Satisfaction Score" DESC LIMIT {int(preview_rows)}'
    with engine.connect() as conn:
        preview = pd.read_sql(text(query), conn)
        n = int(conn.execute(text(f"SELECT COUNT(*) FROM {'`' if engine.dialect.name == 'mysql' else chr(34)}{table}"
                                  f"{'`' if engine.dialect.name == 'mysql' else chr(34)}")).scalar())
        server = conn.execute(text("SELECT VERSION()")).scalar() if engine.dialect.name == "mysql" else "sqlite"
    info = {"exported_at": datetime.now().isoformat(timespec="seconds"), "dialect": engine.dialect.name,
            "server_version": str(server), "table": table, "rows_written": len(frame), "rows_in_table": n,
            "select_query": query, "select_output": json.loads(preview.to_json(orient="records"))}
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(json.dumps(info, indent=2, default=str))
    return info
