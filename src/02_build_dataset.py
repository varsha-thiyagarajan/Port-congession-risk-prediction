"""
STEP 2 - Cleaning, geofences, waiting-time target, hourly table, hotspots.

Input : data/interim/ais_lalb_filtered.parquet, data/raw/weather.csv
Output: data/processed/ais_clean_zones.parquet   cleaned AIS pings + zone label
        data/processed/hourly_table.csv          one row per hour (queue, berth, arrivals, weather)
        data/processed/journeys.csv              one row per ship arrival (port entry -> berth)
        data/processed/anchorage_episodes.csv    one row per stay at anchor
        data/processed/hotspots.csv              DBSCAN anchorage clusters
        data/processed/anchorage_points.parquet  points used for the clusters
        outputs/tables/data_quality.csv          before / after counts
"""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import config as C
from common import assign_zone, vessel_group, size_class, to_local


# ------------------------------------------------------------------ cleaning
def clean(df):
    log = [("Rows in filtered AIS file (cargo + tanker, LA/LB box)", len(df), len(df))]

    def step(name, mask):
        nonlocal df
        before = len(df)
        df = df[mask(df)].copy()
        log.append((name, before - len(df), len(df)))

    step("Invalid / missing position", lambda d: d.latitude.between(-90, 90) & d.longitude.between(-180, 180)
         & ~((d.latitude == 0) & (d.longitude == 0)) & d.latitude.notna() & d.longitude.notna())
    step(f"Missing or impossible speed (missing, <0 or >{C.MAX_SOG_KN:g} kn)",
         lambda d: d.sog.notna() & d.sog.between(0, C.MAX_SOG_KN))
    df = df.sort_values(["mmsi", "base_date_time"])
    step("Duplicate ship + timestamp", lambda d: ~d.duplicated(["mmsi", "base_date_time"], keep="first"))

    # position jumps: implied speed between consecutive pings of one ship
    removed = 0
    for _ in range(5):
        dt_h = df.groupby("mmsi").base_date_time.diff().dt.total_seconds() / 3600
        km = np.hypot(df.groupby("mmsi").latitude.diff() * 111.0,
                      df.groupby("mmsi").longitude.diff() * 111.32 * np.cos(np.radians(33.7)))
        kn = km / dt_h / 1.852
        bad = (kn > C.MAX_JUMP_KN) & (dt_h > 0)
        if not bad.any():
            break
        removed += int(bad.sum())
        df = df[~bad]
    log.append((f"Position jump (implied speed > {C.MAX_JUMP_KN:g} kn)", removed, len(df)))

    # standard fields
    df = df.reset_index(drop=True)
    df["vessel_group"] = vessel_group(df.vessel_type)
    for c in ("length", "width", "draft"):            # static ship data: fill gaps with the ship's own median
        df[c] = df[c].fillna(df.groupby("mmsi")[c].transform("median"))
    df["size_class"] = size_class(df.length)
    dq = pd.DataFrame(log, columns=["step", "rows_removed", "rows_remaining"])
    dq["pct_of_start"] = (dq.rows_remaining / dq.rows_remaining.iloc[0] * 100).round(2)
    return df, dq


# ------------------------------------------------------------- hourly table
def hourly_table(df, weather):
    df = df.copy()
    df["hour"] = df.base_date_time.dt.floor("h")
    hours = pd.date_range(df.hour.min(), df.hour.max(), freq="h")
    anch = df[df.zone == "anchorage"]
    waiting = anch.groupby("hour").mmsi.nunique().reindex(hours, fill_value=0)
    berth = df[df.zone == "berth"].groupby("hour").mmsi.nunique().reindex(hours, fill_value=0)
    rows = df.groupby("hour").size().reindex(hours, fill_value=0)

    sets = anch.groupby("hour").mmsi.agg(set).reindex(hours).apply(lambda s: s if isinstance(s, set) else set())
    arrivals = pd.Series([len(sets.iloc[i] - (sets.iloc[i - 1] if i > 0 else sets.iloc[i]))
                          for i in range(len(sets))], index=hours)

    h = pd.DataFrame({"ships_waiting": waiting, "ships_at_berth": berth,
                      "arrivals": arrivals, "rows_in_hour": rows})
    h["low_coverage"] = h.rows_in_hour < C.LOW_COVERAGE_ROWS
    h.loc[h.low_coverage, ["ships_waiting", "ships_at_berth", "arrivals"]] = np.nan
    h.loc[h.low_coverage.shift(1, fill_value=False), "arrivals"] = np.nan

    # long-stay ships (>= 72 h at anchor) are lay-ups, not queue
    a = anch[["mmsi", "hour"]].drop_duplicates().sort_values(["mmsi", "hour"]).copy()
    a["gap_h"] = a.groupby("mmsi").hour.diff().dt.total_seconds() / 3600
    a["ep"] = (a.gap_h.isna() | (a.gap_h > C.EPISODE_GAP_H)).groupby(a.mmsi).cumsum()
    a["hours_so_far"] = (a.hour - a.groupby(["mmsi", "ep"]).hour.transform("min")).dt.total_seconds() / 3600
    h["long_stay_ships"] = a[a.hours_so_far >= C.LONG_STAY_H].groupby("hour").mmsi.nunique().reindex(hours, fill_value=0)
    h.loc[h.low_coverage, "long_stay_ships"] = np.nan
    h["queue_ships"] = h.ships_waiting - h.long_stay_ships

    local = to_local(h.index)
    h["hour_local"] = local.hour
    h["weekday"] = local.weekday
    h["is_weekend"] = (h.weekday >= 5).astype(int)
    h["date_local"] = local.date
    w = weather.copy()
    w["time"] = pd.to_datetime(w["time"])
    h = h.join(w.set_index("time")[["wave_height", "wind_speed"]], how="left")
    h.index.name = "hour_utc"
    return h


# ------------------------------------------------- anchorage stays + journeys
def anchorage_episodes(df, data_end):
    a = df[df.zone == "anchorage"].sort_values(["mmsi", "base_date_time"]).copy()
    a["gap_h"] = a.groupby("mmsi").base_date_time.diff().dt.total_seconds() / 3600
    a["ep"] = (a.gap_h.isna() | (a.gap_h > C.EPISODE_GAP_H)).groupby(a.mmsi).cumsum()
    eps = a.groupby(["mmsi", "ep"]).agg(start=("base_date_time", "min"), end=("base_date_time", "max"),
                                         lat=("latitude", "mean"), lon=("longitude", "mean"),
                                         vessel_group=("vessel_group", "first"), length=("length", "first"),
                                         draft=("draft", "first")).reset_index()
    eps["hours"] = (eps.end - eps.start).dt.total_seconds() / 3600
    eps["still_at_anchor_at_end"] = eps.end >= data_end - pd.Timedelta(hours=6)
    return eps.drop(columns="ep")


def journeys(df, h):
    """One row per ship arrival: first ping inside the study area -> first ping at a berth."""
    t0, t1 = df.base_date_time.min(), df.base_date_time.max()
    df = df.sort_values(["mmsi", "base_date_time"])
    seg_id = df.groupby("mmsi").base_date_time.diff().dt.total_seconds().div(3600).gt(C.EPISODE_GAP_H) \
        .groupby(df.mmsi).cumsum()
    rows = []
    for (m, s), g in df.groupby([df.mmsi, seg_id]):
        f = g.iloc[0]
        if f.zone != "other" or f.base_date_time <= t0 + pd.Timedelta(hours=1):
            continue                                    # not observed arriving from outside
        zb, za = g[g.zone == "berth"], g[g.zone == "anchorage"]
        t_berth = zb.base_date_time.min() if len(zb) else pd.NaT
        t_anch = za.base_date_time.min() if len(za) else pd.NaT
        anchored_first = pd.notna(t_anch) and (pd.isna(t_berth) or t_anch < t_berth)
        gaps = g.base_date_time.diff().dt.total_seconds().div(3600)
        last = g.iloc[-1]
        if pd.notna(t_berth):
            status = "reached berth"
        elif last.zone == "anchorage" and last.base_date_time >= t1 - pd.Timedelta(hours=6):
            status = "still waiting at end"
        else:
            status = "left / lost"
        rows.append(dict(mmsi=m, arrival_time=f.base_date_time, berth_time=t_berth,
                         anchorage_time=t_anch if anchored_first else pd.NaT,
                         entry_lat=f.latitude, entry_lon=f.longitude, entry_sog=f.sog,
                         vessel_type=f.vessel_type, vessel_group=f.vessel_group, length=f.length,
                         width=f.width, draft=f.draft, size_class=f.size_class,
                         max_gap_h=gaps.max() if len(g) > 1 else 0.0, status=status,
                         last_seen=last.base_date_time, last_zone=last.zone))
    j = pd.DataFrame(rows).sort_values("arrival_time").reset_index(drop=True)
    j["hours_to_berth"] = (j.berth_time - j.arrival_time).dt.total_seconds() / 3600
    j["anchorage_wait_h"] = (j.berth_time - j.anchorage_time).dt.total_seconds() / 3600
    j["gate"] = np.where(j.entry_lat < 33.5, "South", "West")
    direct = j[(j.status == "reached berth") & j.anchorage_time.isna()].hours_to_berth
    j.attrs["typical_transit_h"] = float(direct.median())
    j["typical_transit_h"] = j.attrs["typical_transit_h"]
    j["excess_delay_h"] = (j.hours_to_berth - j.typical_transit_h).clip(lower=0)
    j["delayed"] = (j.hours_to_berth > C.DELAY_THRESHOLD_H).astype("Int64").where(j.status == "reached berth")
    j["has_gap"] = j.max_gap_h > 6
    j["hour_local"] = to_local(j.arrival_time).hour
    j["date_local"] = to_local(j.arrival_time).date
    # situation in the port at the moment the ship arrived (known at that time)
    hh = h.reset_index().rename(columns={"hour_utc": "hr"})
    j["hr"] = j.arrival_time.dt.floor("h")
    j = j.merge(hh[["hr", "queue_ships", "ships_at_berth", "wave_height", "wind_speed"]]
                .rename(columns={"queue_ships": "queue_at_arrival", "ships_at_berth": "berth_at_arrival"}),
                on="hr", how="left").drop(columns="hr")
    return j


# ------------------------------------------------------------------ hotspots
def hotspots(df, min_ships=3):
    """DBSCAN on stationary anchorage positions. A 'hotspot' needs at least `min_ships` different ships;
    smaller clusters are single-ship stays (typically lay-ups) and are labelled as such."""
    an = df[df.zone == "anchorage"].copy()
    an["hour"] = an.base_date_time.dt.floor("h")
    km_x = lambda lon: lon * np.cos(np.radians(33.7)) * 111.32
    km_y = lambda lat: lat * 110.57
    pts = (an.assign(lat_r=an.latitude.round(4), lon_r=an.longitude.round(4))
             .drop_duplicates(["mmsi", "lat_r", "lon_r"]).copy())          # one point per ship per spot
    pts["cluster"] = DBSCAN(eps=C.DBSCAN_EPS_KM, min_samples=C.DBSCAN_MIN_PTS) \
        .fit_predict(np.c_[km_x(pts.longitude), km_y(pts.latitude)])
    summ = (pts[pts.cluster >= 0].groupby("cluster")
            .agg(positions=("mmsi", "size"), ships=("mmsi", "nunique"),
                 lat=("latitude", "mean"), lon=("longitude", "mean")).reset_index()
            .sort_values("ships", ascending=False).reset_index(drop=True))
    multi = summ[summ.ships >= min_ships].index
    summ["hotspot"] = [f"Hotspot {chr(65 + i)}" if i in multi else "Single-ship spot" for i in range(len(summ))]
    summ["is_hotspot"] = summ.ships >= min_ships
    pts["hotspot"] = pts.cluster.map(dict(zip(summ.cluster, summ.hotspot))).fillna("Not clustered")

    # ship-hours at anchor: assign each anchorage ping to the closest cluster centre (within 1 km)
    d = np.hypot(km_x(an.longitude.values)[:, None] - km_x(summ.lon.values)[None, :],
                 km_y(an.latitude.values)[:, None] - km_y(summ.lat.values)[None, :])
    near = d.min(axis=1) <= 1.0
    an["cluster"] = np.where(near, summ.cluster.values[d.argmin(axis=1)], -1)
    sh = an.drop_duplicates(["mmsi", "hour"]).groupby("cluster").size()
    summ["ship_hours"] = summ.cluster.map(sh).fillna(0).astype(int)
    total = an.drop_duplicates(["mmsi", "hour"]).shape[0]
    summ["share_of_anchorage_ship_hours_pct"] = (summ.ship_hours / total * 100).round(1)
    return summ, pts


def main():
    raw = pd.read_parquet(C.AIS_FILTERED)
    df, dq = clean(raw)
    df["zone"] = assign_zone(df)
    dq.loc[len(dq)] = ["Final analysis table (ping level)", 0, len(df), round(len(df) / len(raw) * 100, 2)]
    extraction = C.TABLES / "extraction_log.csv"
    if extraction.exists():
        print("Raw-file extraction log found:", extraction)
    dq.to_csv(C.TABLES / "data_quality.csv", index=False)
    print(dq.to_string(index=False))

    weather = pd.read_csv(C.WEATHER_CSV)
    h = hourly_table(df, weather)
    data_end = df.base_date_time.max()
    eps = anchorage_episodes(df, data_end)
    j = journeys(df, h)
    summ, pts = hotspots(df)

    df.to_parquet(C.PROCESSED / "ais_clean_zones.parquet")
    h.to_csv(C.PROCESSED / "hourly_table.csv")
    j.to_csv(C.PROCESSED / "journeys.csv", index=False)
    eps.to_csv(C.PROCESSED / "anchorage_episodes.csv", index=False)
    summ.to_csv(C.PROCESSED / "hotspots.csv", index=False)
    pts[["mmsi", "latitude", "longitude", "hotspot"]].to_parquet(C.PROCESSED / "anchorage_points.parquet")

    print(f"\nPeriod: {df.base_date_time.min()} -> {data_end} | ships: {df.mmsi.nunique()}")
    print(f"Hourly rows: {len(h)} (low coverage: {int(h.low_coverage.sum())})")
    print(f"Journeys: {len(j)} | reached berth: {(j.status == 'reached berth').sum()} | "
          f"delayed (> {C.DELAY_THRESHOLD_H:g} h): {int(j.delayed.fillna(0).sum())} | "
          f"typical direct transit: {j.typical_transit_h.iloc[0]:.2f} h")
    print("Hotspots:\n", summ.to_string(index=False))


if __name__ == "__main__":
    main()
