"""The 14 story charts. Plain chart types on purpose: map, bars, lines, one heatmap.
Every function takes the dict from data_access.load_all() (+ a few computed extras) and returns a Figure."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr
import config as C
from common import to_local, congestion_class

NAVY, TEAL, ORANGE, AMBER, GREY, RED, GREEN = "#1f3a5f", "#2a9d8f", "#e76f51", "#e9c46a", "#8d99ae", "#d62828", "#2a9d8f"
CLS_COL = {"Low": GREEN, "Medium": AMBER, "High": ORANGE}
GROUP_COL = {"Cargo": NAVY, "Tanker": ORANGE}

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10, "axes.titleweight": "bold",
                     "axes.titlesize": 12, "axes.titlelocation": "left", "figure.facecolor": "white"})


def _boxes(ax, labels=False):
    for b, c, n in [(C.BERTH, NAVY, "Berths"), (C.BERTH_W, NAVY, None), (C.ANCH_IN, RED, "Anchorage"), (C.ANCH_OUT, RED, None)]:
        ax.add_patch(Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2], fill=False, edgecolor=c, lw=1.5))


def _map_axes(ax):
    ax.set_xlim(-118.35, -117.95); ax.set_ylim(33.58, 33.80)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_aspect(1 / np.cos(np.radians(33.7)))


def _dates(ax, fmt="%b %d"):
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))


def _sub(fig, text):
    fig.text(0.01, 0.005, text, fontsize=8, color=GREY, ha="left", va="bottom")


# ---------------------------------------------------------------- V01
def v01_zone_map(d):
    ais = d["ais"]
    fig, ax = plt.subplots(figsize=(9, 6.2))
    s = ais.sample(min(len(ais), 60000), random_state=1)
    ax.scatter(s.longitude, s.latitude, s=1, c="#d0d4da", label="All AIS positions")
    for z, c, n in [("anchorage", RED, "Stationary at anchorage"), ("berth", NAVY, "Stationary at berth")]:
        p = ais[ais.zone == z]; p = p.sample(min(len(p), 20000), random_state=1)
        ax.scatter(p.longitude, p.latitude, s=3, c=c, label=n)
    _boxes(ax); _map_axes(ax)
    ax.set_title("V01  Study area: where ships wait and where they berth")
    ax.legend(markerscale=4, loc="lower left", fontsize=8)
    fig.tight_layout(); return fig


# ---------------------------------------------------------------- V02
def v02_traffic_density(d):
    mv = d["ais"].query("sog >= @C.STATIONARY_KN")
    fig, ax = plt.subplots(figsize=(9, 6.2))
    hb = ax.hexbin(mv.longitude, mv.latitude, gridsize=110, bins="log", cmap="viridis", mincnt=1,
                   extent=(-118.6, -117.9, 33.4, 33.85))
    _boxes(ax); ax.grid(False)
    ax.set_xlim(-118.6, -117.9); ax.set_ylim(33.4, 33.85); ax.set_aspect(1 / np.cos(np.radians(33.7)))
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    fig.colorbar(hb, label="AIS messages from moving ships (log scale)", shrink=0.8)
    ax.set_title("V02  Traffic density: the approach lanes into the port")
    fig.tight_layout(); return fig


# ---------------------------------------------------------------- V03
def v03_hotspots(d):
    ais, pts, hs = d["ais"], d["anchor_pts"], d["hotspots"]
    fig, ax = plt.subplots(figsize=(9, 6.2))
    s = ais.sample(min(len(ais), 40000), random_state=1)
    ax.scatter(s.longitude, s.latitude, s=1, c="#e3e6ea")
    ax.scatter(pts[pts.hotspot.isin(["Not clustered"]) | pts.hotspot.eq("Single-ship spot")].longitude,
               pts[pts.hotspot.isin(["Not clustered"]) | pts.hotspot.eq("Single-ship spot")].latitude, s=6, c=GREY, label="Single-ship / not clustered")
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(hs[hs.is_hotspot].itertuples()):
        p = pts[pts.hotspot == r.hotspot]
        ax.scatter(p.longitude, p.latitude, s=9, color=cmap(i), label=f"{r.hotspot}: {r.ships} ships, {r.share_of_anchorage_ship_hours_pct:.0f}% of anchor hours")
    _boxes(ax); _map_axes(ax)
    ax.set_title("V03  Congestion hotspots found by DBSCAN")
    ax.legend(fontsize=7, loc="lower left", ncol=1)
    fig.tight_layout(); return fig


# ---------------------------------------------------------------- V04
def v04_queue_timeline(d, q1, q2):
    h = d["hourly"]; loc = to_local(h.index)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    top = max(h.queue_ships.max(), q2) + 2
    ax.axhspan(0, q1, color=GREEN, alpha=0.08); ax.axhspan(q1, q2, color=AMBER, alpha=0.12); ax.axhspan(q2, top, color=ORANGE, alpha=0.10)
    first_valid = loc[0] + pd.Timedelta(hours=C.LONG_STAY_H)
    pre = loc < first_valid
    ax.plot(loc[~pre | (np.arange(len(loc)) == pre.sum())], h.queue_ships[~pre | (np.arange(len(loc)) == pre.sum())], color=NAVY, lw=1.6, label="Ships queueing (at anchor < 3 days)")
    ax.plot(loc[pre], h.queue_ships[pre], color=NAVY, lw=1.2, alpha=.3, ls="--", label="Warm-up (not used)")
    ax.plot(loc[~pre], h.long_stay_ships[~pre], color=GREY, lw=1, label="Long-stay ships (3+ days, excluded)")
    ax.fill_between(loc, 0, top, where=h.low_coverage.values, color="grey", alpha=0.3, label="Low AIS coverage")
    ax.axvline(first_valid, color="k", ls=":", lw=1); ax.text(first_valid, top - 0.3, " warm-up ends", fontsize=8, va="top")
    _dates(ax)
    ax.text(loc[1], q1 / 2, "Low", color=GREEN, fontsize=9, va="center"); ax.text(loc[1], (q1 + q2) / 2, "Medium", color="#b08900", fontsize=9, va="center")
    ax.text(loc[1], (q2 + top) / 2, "High", color=ORANGE, fontsize=9, va="center")
    ax.set_ylim(0, top); ax.set_ylabel("Ships"); ax.set_xlabel("Local time (Pacific)")
    ax.set_title("V04  The queue at LA / Long Beach, hour by hour")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout(); return fig


# ---------------------------------------------------------------- V05
def v05_journey_distribution(d):
    j = d["journeys"]; r = j[j.status == "reached berth"]
    typ = r.typical_transit_h.iloc[0]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bins = np.logspace(np.log10(2), np.log10(r.hours_to_berth.max() * 1.05), 26)      # equal-width on a log axis
    direct = r[r.delayed == 0].hours_to_berth; delayed = r[r.delayed == 1].hours_to_berth
    ax.hist(direct, bins=bins, color=TEAL, label=f"On time ({len(direct)} ships)")
    ax.hist(delayed, bins=bins, color=ORANGE, label=f"Delayed > {C.DELAY_THRESHOLD_H:g} h ({len(delayed)} ships)")
    ax.set_xscale("log"); ticks = [t for t in [2, 4, 6, 12, 24, 48, 96] if t <= bins[-1]]
    ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks]); ax.minorticks_off()
    ax.axvline(typ, color="k", ls="--", lw=1); ax.text(typ * 1.04, ax.get_ylim()[1] * 0.92, f"typical direct trip {typ:.1f} h", fontsize=8)
    ax.set_xlabel("Hours from entering the port area to reaching a berth (log scale)"); ax.set_ylabel("Number of ships")
    ax.set_title("V05  Most ships glide in; a minority get stuck for days")
    ax.legend(); fig.tight_layout(); return fig


# ---------------------------------------------------------------- V06
def v06_delay_by_vessel(d, fisher_p):
    r = d["journeys"].query("status == 'reached berth'")
    g = r.groupby("vessel_group").agg(n=("delayed", "size"), delayed=("delayed", "sum"), med_wait=("excess_delay_h", "median"))
    g["share"] = g.delayed / g.n * 100
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    b = ax.bar(g.index, g.share, color=[GROUP_COL[i] for i in g.index], width=0.55)
    for i, (k, row) in enumerate(g.iterrows()):
        ax.text(i, row.share + 1, f"{row.share:.0f}%\n({int(row.delayed)} of {int(row.n)})", ha="center", fontsize=9)
    ax.set_ylim(0, g.share.max() * 1.35); ax.set_ylabel("Share of arrivals delayed > 6 h (%)")
    ax.set_title("V06  Tankers are delayed far more often than cargo ships")
    ax.text(0.98, 0.95, f"Fisher exact test p = {fisher_p:.3f}", transform=ax.transAxes, ha="right", fontsize=9)
    fig.tight_layout(); return fig


# ---------------------------------------------------------------- V07
def v07_arrivals_by_hour(d):
    j = d["journeys"]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    hrs = np.arange(24)
    allc = j.groupby("hour_local").size().reindex(hrs, fill_value=0)
    dl = j[j.delayed == 1].groupby("hour_local").size().reindex(hrs, fill_value=0)
    ax.bar(hrs, allc - dl, color=TEAL, label="Arrivals")
    ax.bar(hrs, dl, bottom=allc - dl, color=ORANGE, label="…of which delayed > 6 h")
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks(range(0, 24, 2)); ax.set_xlabel("Hour ships entered the port area (local Pacific time)"); ax.set_ylabel("Number of ships")
    ax.set_title("V07  When ships arrive: pilots and tugs are needed in waves")
    ax.legend(); fig.tight_layout(); return fig


# ---------------------------------------------------------------- V08
def v08_queue_by_hour(d, kw_p):
    v = d["valid"]
    g = v.groupby("hour_local").queue_ships
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.fill_between(g.median().index, g.quantile(.25), g.quantile(.75), color=NAVY, alpha=.2, label="Middle 50% of hours")
    ax.plot(g.median().index, g.median(), color=NAVY, marker="o", label="Median queue")
    ax.set_xticks(range(0, 24, 2)); ax.set_xlabel("Hour of day (local Pacific time)"); ax.set_ylabel("Ships in queue")
    ax.set_title("V08  The queue barely changes with time of day")
    ax.text(0.99, 0.04, f"Kruskal-Wallis across 4 time blocks: p = {kw_p:.2f} (not significant)", transform=ax.transAxes, ha="right", fontsize=9)
    ax.legend(loc="upper left"); fig.tight_layout(); return fig


# ---------------------------------------------------------------- V09
def v09_spearman(d):
    v = d["valid"]
    cols = ["queue_ships", "ships_at_berth", "arrivals", "long_stay_ships", "wave_height", "wind_speed"]
    lab = ["Queue", "Ships at berth", "New at anchor", "Long-stay ships", "Wave height", "Wind speed"]
    x = v[cols].dropna()
    t = np.arange(len(x))
    det = x.apply(lambda s: s - np.polyval(np.polyfit(t, s.values, 1), t))
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"wspace": 0.08})
    for ax, data, title in [(axs[0], x, "Raw"), (axs[1], det, "After removing the 2-week trend")]:
        c = data.corr(method="spearman").round(2)
        im = ax.imshow(c, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(lab, rotation=40, ha="right")
        ax.set_yticks(range(len(cols))); ax.set_yticklabels(lab if ax is axs[0] else []); ax.grid(False)
        for i in range(len(cols)):
            for k in range(len(cols)):
                ax.text(k, i, f"{c.iloc[i, k]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(title, fontsize=10)
    fig.suptitle("V09  What moves with the queue? (Spearman correlation, hourly)", x=0.01, ha="left", fontweight="bold")
    fig.colorbar(im, ax=axs, shrink=0.8)
    return fig


# ---------------------------------------------------------------- V10
def v10_daily_queue(d, q2):
    v = d["valid"]
    g = v.groupby("date_local").agg(mean=("queue_ships", "mean"), high=("queue_ships", lambda s: (s > q2).mean() * 100))
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    x = [str(i)[5:] for i in g.index]
    ax.bar(x, g["mean"], color=NAVY, alpha=.85)
    for i, (m, hh) in enumerate(zip(g["mean"], g["high"])):
        ax.text(i, m + .15, f"{m:.1f}", ha="center", fontsize=8)
    ax2 = ax.twinx(); ax2.plot(x, g["high"], color=ORANGE, marker="o", label="% of hours in High"); ax2.set_ylim(0, 105); ax2.grid(False)
    ax2.set_ylabel("% of the day in High congestion", color=ORANGE); ax2.spines["right"].set_visible(True)
    ax.set_ylabel("Average ships in queue"); ax.set_xlabel("Day (month-day, local time)")
    ax.set_title("V10  Congestion comes in multi-day swings, not hourly ones")
    fig.tight_layout(); return fig


# ---------------------------------------------------------------- V11
def v11_demurrage(d, ship_hours, days):
    rates = C.DEMURRAGE_RATES_USD_DAY
    tot = np.array([ship_hours / 24 * r for r in rates]) / 1e6
    fig, ax = plt.subplots(figsize=(8.5, 4.6)); xx = np.arange(len(rates)); w = .27
    ax.bar(xx - w, tot, w, color=NAVY, label="Cost of the queue in the period")
    for k, (red, col) in enumerate(zip(C.REDUCTION_SCENARIOS, [TEAL, "#8fd3c7"])):
        ax.bar(xx + k * w, tot * red, w, color=col, label=f"Avoided if queue cut by {red*100:.0f}%")
    for i, t in enumerate(tot): ax.text(i - w, t + .02, f"${t:.1f}M", ha="center", fontsize=8)
    ax.set_xticks(xx); ax.set_xticklabels([f"${r:,}/day" for r in rates]); ax.set_xlabel("ASSUMED demurrage rate per ship (not from data)")
    ax.set_ylabel("Million USD")
    ax.set_title(f"V11  Scenario: {ship_hours:,.0f} ship-hours in the queue over {days:.0f} days")
    ax.legend(fontsize=8); fig.tight_layout(); return fig


# ---------------------------------------------------------------- V12
def v12_model_vs_naive(d):
    r = d["model_results"]
    pers = r[r.model == "Baseline: same as now"].set_index("horizon_h").MAE
    fin = r[r.model.str.startswith("Final")].set_index("horizon_h").MAE
    hs = sorted(pers.index)
    fig, ax = plt.subplots(figsize=(8.5, 4.6)); xx = np.arange(len(hs)); w = .36
    ax.bar(xx - w / 2, pers.loc[hs], w, color=GREY, label="Naive: queue stays as it is now")
    ax.bar(xx + w / 2, fin.loc[hs], w, color=TEAL, label="Machine-learning blend")
    for i, k in enumerate(hs):
        better = (1 - fin[k] / pers[k]) * 100
        ax.text(i, max(pers[k], fin[k]) + .08, f"{better:+.0f}%", ha="center", fontsize=9, fontweight="bold", color=TEAL if better > 0 else RED)
    ax.set_xticks(xx); ax.set_xticklabels([f"{k} h ahead" for k in hs]); ax.set_ylabel("Average error (ships), lower is better")
    ax.set_title("V12  The model earns its keep at day-ahead, not hour-ahead")
    ax.legend(fontsize=8, loc="upper left"); fig.tight_layout(); return fig


# ---------------------------------------------------------------- V13
def v13_actual_vs_pred(d, q1, q2, H):
    o = d["predictions_oof"].query("horizon_h == @H").copy()
    o["origin_utc"] = pd.to_datetime(o["origin_utc"]); o["target"] = to_local(o.origin_utc + pd.Timedelta(hours=H))
    o = o.sort_values("target")
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(o.target, o.actual, color="k", lw=2, label="Actual queue")
    ax.plot(o.target, o.persistence, color=GREY, lw=1.2, label=f"Naive: queue {H} h earlier")
    ax.plot(o.target, o.blend, color=TEAL, lw=1.8, ls="--", label=f"Model forecast made {H} h earlier")
    ax.axhline(q2, color=ORANGE, ls=":", lw=1); ax.text(o.target.iloc[0], q2 + .15, f"High above {q2:.0f}", color=ORANGE, fontsize=8)
    ax.set_ylabel("Ships in queue"); ax.set_xlabel("Local time (Pacific)")
    _dates(ax, "%b %d %H:%M")
    ax.set_title(f"V13  Day-ahead forecast vs reality (never-seen test days)")
    ax.legend(fontsize=8, ncol=3, loc="upper left"); fig.tight_layout(); return fig


# ---------------------------------------------------------------- V14
def v14_high_caught(d, H):
    c = d["classification_results"].query("horizon_h == @H")
    keep = ["Baseline: class of queue now", "Class from blend forecast", "Logistic Regression classifier", "LightGBM classifier"]
    c = c[c.model.isin(keep)].set_index("model").loc[keep]
    fig, ax = plt.subplots(figsize=(9, 4.6)); xx = np.arange(len(c)); w = .38
    ax.bar(xx - w / 2, c.high_caught, w, color=TEAL, label="High-congestion hours caught")
    ax.bar(xx + w / 2, c.false_alarms, w, color=ORANGE, label="False alarms")
    tot = int(c.n_high_hours.iloc[0]); ax.axhline(tot, color="k", ls="--", lw=1); ax.text(len(c) - .5, tot + .5, f"all High hours in test = {tot}", ha="right", fontsize=8)
    ax.set_xticks(xx); ax.set_xticklabels([m.replace(": ", ":\n").replace(" classifier", "\nclassifier") for m in c.index], fontsize=8)
    ax.set_ylabel("Hours"); ax.set_title(f"V14  Early warning {H} h ahead: how many High hours are caught?")
    ax.set_ylim(0, tot * 1.25); ax.legend(fontsize=8, loc="upper left", ncol=2); fig.tight_layout(); return fig
