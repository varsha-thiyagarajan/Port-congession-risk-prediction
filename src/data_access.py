"""One place that loads every processed table (used by analytics, charts and the dashboard)."""
import joblib
import pandas as pd
import config as C


def load_all(with_ais=True):
    d = {}
    d["hourly"] = pd.read_csv(C.PROCESSED / "hourly_table.csv", index_col=0, parse_dates=True)
    d["journeys"] = pd.read_csv(C.PROCESSED / "journeys.csv",
                                parse_dates=["arrival_time", "berth_time", "anchorage_time", "last_seen"])
    d["episodes"] = pd.read_csv(C.PROCESSED / "anchorage_episodes.csv", parse_dates=["start", "end"])
    d["hotspots"] = pd.read_csv(C.PROCESSED / "hotspots.csv")
    d["dq"] = pd.read_csv(C.TABLES / "data_quality.csv")
    if with_ais:
        d["ais"] = pd.read_parquet(C.PROCESSED / "ais_clean_zones.parquet")
        d["anchor_pts"] = pd.read_parquet(C.PROCESSED / "anchorage_points.parquet")
    for name in ("model_results", "classification_results", "feature_ablation", "predictions_oof", "stats_tests"):
        f = C.TABLES / f"{name}.csv"
        if f.exists():
            d[name] = pd.read_csv(f)
    return d


def valid_hours(h):
    """Hours usable for queue statistics: after the 72 h warm-up and with good AIS coverage."""
    ok = (h.index >= h.index.min() + pd.Timedelta(hours=C.LONG_STAY_H)) & h.queue_ships.notna()
    return h[ok].copy()


def load_models():
    out = {}
    for H in C.HORIZONS:
        f = C.MODELS / f"queue_forecast_{H}h.joblib"
        if f.exists():
            out[H] = joblib.load(f)
    return out
