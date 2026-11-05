"""Small helpers shared by the pipeline, the models and the dashboard."""
import numpy as np
import pandas as pd
import config as C


def inside(df, box):
    return df.longitude.between(box[0], box[1]) & df.latitude.between(box[2], box[3])


def assign_zone(df):
    """anchorage / berth / other. Only stationary ships (sog < 1 kn) are counted."""
    slow = df.sog < C.STATIONARY_KN
    berth_box = inside(df, C.BERTH) | inside(df, C.BERTH_W)
    anch_box = inside(df, C.ANCH_IN) | inside(df, C.ANCH_OUT)
    zone = pd.Series("other", index=df.index)
    zone[anch_box & slow] = "anchorage"
    zone[berth_box & slow] = "berth"
    # where the two boxes overlap, an AIS "at anchor" status (1) wins
    zone[berth_box & anch_box & slow & (df.status == 1)] = "anchorage"
    return zone


def vessel_group(vessel_type):
    """AIS ship-type codes 70-79 = cargo, 80-89 = tanker."""
    return np.where(vessel_type >= 80, "Tanker", "Cargo")


def size_class(length):
    return pd.cut(length, [0, 200, 300, 1000],
                  labels=["Small (<200 m)", "Medium (200-300 m)", "Large (>300 m)"])


def to_local(ts):
    """AIS time is UTC. Port operations run on Pacific time."""
    ts = pd.DatetimeIndex(ts)
    return ts.tz_localize("UTC").tz_convert(C.TZ_LOCAL).tz_localize(None)


def congestion_thresholds(queue_series):
    """Low / Medium / High cut-offs = terciles of the training queue."""
    q1, q2 = queue_series.quantile([1 / 3, 2 / 3])
    return float(q1), float(q2)


def congestion_class(values, q1, q2):
    v = np.asarray(values, dtype=float)
    return np.where(v <= q1, "Low", np.where(v <= q2, "Medium", "High"))
