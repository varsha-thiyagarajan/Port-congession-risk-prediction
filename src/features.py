"""Feature builder used by training AND by the dashboard (so both always agree).

One row = one forecast ORIGIN hour T. Everything in the row is known at T.
Target = queue size at T + H hours.
"""
import numpy as np
import pandas as pd
import config as C

# Final model inputs: the queue's own recent history + time of day of the forecast hour.
# Berth count, arrivals, weather and weekday were tested (see feature_ablation.csv) and did NOT improve
# forecasts on this 2-week dataset, so they are not used by the final model.
CORE = ["queue_now", "queue_3h_ago", "queue_6h_ago", "queue_mean_24h", "queue_std_24h", "queue_trend_3h",
        "target_hour_sin", "target_hour_cos"]
PORT_STATE = ["ships_at_berth", "arrivals_last_6h", "arrivals_last_24h", "long_stay_ships"]
CALENDAR = ["target_weekday"]
WEATHER = ["wave_height_now", "wind_speed_now"]
ALL_FEATURES = CORE + PORT_STATE + CALENDAR + WEATHER
FEATURES = CORE

FEATURE_LABELS = {
    "queue_now": "Queue now", "queue_3h_ago": "Queue 3 h ago", "queue_6h_ago": "Queue 6 h ago",
    "queue_mean_24h": "Average queue, last 24 h", "queue_std_24h": "Queue variability, last 24 h",
    "queue_trend_3h": "Queue change, last 3 h", "ships_at_berth": "Ships at berth now",
    "arrivals_last_6h": "New ships at anchor, last 6 h", "arrivals_last_24h": "New ships at anchor, last 24 h",
    "long_stay_ships": "Long-stay (3+ days) ships", "target_hour_sin": "Time of day (sin)",
    "target_hour_cos": "Time of day (cos)", "target_weekday": "Weekday of forecast hour",
    "wave_height_now": "Wave height now (m)", "wind_speed_now": "Wind speed now",
}


def load_hourly():
    h = pd.read_csv(C.PROCESSED / "hourly_table.csv", index_col=0, parse_dates=True)
    return h


def build(h, horizon, with_target=True):
    """Return the feature table indexed by origin hour (UTC)."""
    hi = h.copy()
    for c in ("queue_ships", "ships_at_berth", "arrivals", "long_stay_ships"):
        hi[c] = hi[c].interpolate(limit=6)                       # bridge short coverage gaps (features only)
    q = hi.queue_ships
    f = pd.DataFrame(index=h.index)
    f["queue_now"] = q
    f["queue_3h_ago"] = q.shift(3)
    f["queue_6h_ago"] = q.shift(6)
    f["queue_mean_24h"] = q.rolling(24, min_periods=12).mean()
    f["queue_std_24h"] = q.rolling(24, min_periods=12).std()
    f["queue_trend_3h"] = f.queue_now - f.queue_3h_ago
    f["ships_at_berth"] = hi.ships_at_berth
    f["arrivals_last_6h"] = hi.arrivals.rolling(6, min_periods=4).sum()
    f["arrivals_last_24h"] = hi.arrivals.rolling(24, min_periods=12).sum()
    f["long_stay_ships"] = hi.long_stay_ships
    tgt_local = (h.index + pd.Timedelta(hours=horizon)).tz_localize("UTC").tz_convert(C.TZ_LOCAL)
    f["target_hour_sin"] = np.sin(2 * np.pi * tgt_local.hour / 24)
    f["target_hour_cos"] = np.cos(2 * np.pi * tgt_local.hour / 24)
    f["target_weekday"] = tgt_local.weekday
    f["wave_height_now"] = h.wave_height
    f["wind_speed_now"] = h.wind_speed
    # the first 72 h of the file cannot tell a long-stay ship from a queueing one
    warm = h.index >= h.index.min() + pd.Timedelta(hours=C.LONG_STAY_H)
    f = f[warm]
    if with_target:
        f["y"] = h.queue_ships.shift(-horizon).reindex(f.index)
        f = f.dropna(subset=ALL_FEATURES + ["y"])
    else:
        f = f.dropna(subset=ALL_FEATURES)
    return f
