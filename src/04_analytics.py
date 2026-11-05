"""
STEP 4 - Analytics: statistics, KPIs and the 14 story charts.

Output: outputs/charts/V01...V14 *.png
        outputs/tables/stats_tests.csv, visual_insights.csv, kpis.json, decision_table.csv, demurrage_scenarios.csv
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, kruskal, mannwhitneyu, spearmanr
import matplotlib.pyplot as plt
import config as C
import charts as ch
from common import congestion_thresholds, to_local
from data_access import load_all, load_models, valid_hours
from decisions import decision_table


def kw_effect(stat, groups):
    n, k = sum(len(g) for g in groups), len(groups)
    return max(0.0, (stat - k + 1) / (n - k))        # epsilon-squared


def verdict(p):
    return "significant (p < 0.05)" if p < 0.05 else "not significant"


def run_tests(v, r):
    rows = []
    # time of day (local) - Kruskal-Wallis
    blk = pd.cut(v.hour_local, [-1, 5, 11, 17, 23], labels=["Night 00-05", "Morning 06-11", "Afternoon 12-17", "Evening 18-23"])
    groups = [v.queue_ships[blk == b].values for b in blk.cat.categories]
    s, p = kruskal(*groups); kw_time_p = p
    rows.append(("Kruskal-Wallis", "Queue size differs between 4 time-of-day blocks?", len(v), s, p, kw_effect(s, groups), "epsilon^2"))
    # weekday vs weekend
    a, b = v.queue_ships[v.is_weekend == 0], v.queue_ships[v.is_weekend == 1]
    s, p = mannwhitneyu(a, b); rows.append(("Mann-Whitney", "Queue size differs weekday vs weekend? (only 2 weekends of data)", len(v), s, p, 1 - 2 * s / (len(a) * len(b)), "rank-biserial"))
    # berth level (same hour, diagnostic)
    lvl = pd.qcut(v.ships_at_berth.rank(method="first"), 3, labels=["Low", "Medium", "High"])
    groups = [v.queue_ships[lvl == l].values for l in ["Low", "Medium", "High"]]
    s, p = kruskal(*groups); rows.append(("Kruskal-Wallis", "Queue size differs by berth occupancy level (same hour, diagnostic only)", len(v), s, p, kw_effect(s, groups), "epsilon^2"))
    # Spearman raw + detrended
    x = v[["queue_ships", "ships_at_berth", "arrivals", "wave_height", "wind_speed"]].dropna()
    t = np.arange(len(x)); det = x.apply(lambda c: c - np.polyval(np.polyfit(t, c.values, 1), t))
    for col, name in [("ships_at_berth", "berth occupancy"), ("arrivals", "new ships at anchor"), ("wave_height", "wave height"), ("wind_speed", "wind speed")]:
        rho, p = spearmanr(x[col], x.queue_ships); rows.append(("Spearman", f"Queue vs {name} (raw)", len(x), np.nan, p, rho, "rho"))
        rho, p = spearmanr(det[col], det.queue_ships); rows.append(("Spearman", f"Queue vs {name} (after removing 2-week trend)", len(x), np.nan, p, rho, "rho"))
    # vessel group
    ct = pd.crosstab(r.vessel_group, r.delayed)
    odds, fisher_p = fisher_exact([[ct.loc["Tanker", 1], ct.loc["Tanker", 0]], [ct.loc["Cargo", 1], ct.loc["Cargo", 0]]])
    rows.append(("Fisher exact", "Tankers delayed > 6 h more often than cargo ships?", len(r), np.nan, fisher_p, odds, "odds ratio"))
    s, p = mannwhitneyu(r[r.vessel_group == "Tanker"].hours_to_berth, r[r.vessel_group == "Cargo"].hours_to_berth)
    rows.append(("Mann-Whitney", "Hours to berth differ tanker vs cargo?", len(r), s, p, np.nan, ""))
    groups = [g.hours_to_berth.values for _, g in r.groupby("size_class", observed=True)]
    s, p = kruskal(*groups); rows.append(("Kruskal-Wallis", "Hours to berth differ by ship size class?", len(r), s, p, kw_effect(s, groups), "epsilon^2"))
    s, p = mannwhitneyu(r[r.gate == "West"].hours_to_berth, r[r.gate == "South"].hours_to_berth)
    rows.append(("Mann-Whitney", "Hours to berth differ by approach (West vs South)?", len(r), s, p, np.nan, ""))
    out = pd.DataFrame(rows, columns=["test", "question", "n", "statistic", "p_value", "effect_size", "effect_measure"])
    out["conclusion"] = out.p_value.map(verdict)
    return out.round(4), kw_time_p, fisher_p


def peak_window(j, width=4):
    """The busiest `width`-hour window of the day for new arrivals (local time, wraps midnight)."""
    c = j.groupby("hour_local").size().reindex(range(24), fill_value=0)
    sums = {s: sum(c[(s + k) % 24] for k in range(width)) for s in range(24)}
    start = max(sums, key=sums.get)
    return dict(arrival_peak_start_local=int(start), arrival_peak_end_local=int((start + width) % 24),
                arrival_peak_share_pct=float(sums[start] / c.sum() * 100))


def main():
    d = load_all()
    h, j = d["hourly"], d["journeys"]
    v = valid_hours(h); d["valid"] = v
    r = j[j.status == "reached berth"].copy()
    models = load_models()
    # one definition of Low / Medium / High everywhere: the one the final forecast model was trained with
    q1, q2 = (models[C.FINAL_HORIZON]["q1"], models[C.FINAL_HORIZON]["q2"]) if models else congestion_thresholds(v.queue_ships)

    tests, kw_time_p, fisher_p = run_tests(v, r)
    tests.to_csv(C.TABLES / "stats_tests.csv", index=False)
    print(tests[["test", "question", "p_value", "effect_size", "conclusion"]].to_string(index=False))

    # ---------------- KPIs (everything computed, nothing typed in) ----------------
    ais = d["ais"]
    dl, ok = r[r.delayed == 1], r[r.delayed == 0]
    ship_hours = float(v.queue_ships.sum()); days = len(v) / 24
    peak_h = v.queue_ships.idxmax()
    tan = r[r.vessel_group == "Tanker"].delayed.mean() * 100; car = r[r.vessel_group == "Cargo"].delayed.mean() * 100
    kpi = dict(
        period_start=str(ais.base_date_time.min()), period_end=str(ais.base_date_time.max()),
        ships=int(ais.mmsi.nunique()), arrivals_observed=int(len(j)), arrivals_reached_berth=int(len(r)),
        delayed_arrivals=int(len(dl)), delayed_share_pct=float(len(dl) / len(r) * 100),
        typical_direct_trip_h=float(r.typical_transit_h.iloc[0]),
        median_hours_to_berth_delayed=float(dl.hours_to_berth.median()), max_hours_to_berth=float(r.hours_to_berth.max()),
        delayed_share_of_excess_hours_pct=float(dl.excess_delay_h.sum() / r.excess_delay_h.sum() * 100),
        tanker_delayed_pct=float(tan), cargo_delayed_pct=float(car), fisher_p=float(fisher_p),
        queue_mean=float(v.queue_ships.mean()), queue_median=float(v.queue_ships.median()), queue_peak=float(v.queue_ships.max()),
        queue_peak_time_local=str(to_local([peak_h])[0]), q1=q1, q2=q2,
        high_hours_pct=float((v.queue_ships > q2).mean() * 100), valid_hours=int(len(v)), days=float(days),
        queue_ship_hours=ship_hours, still_waiting_at_end=int((j.status == "still waiting at end").sum()),
        low_coverage_hours=int(h.low_coverage.sum()), long_stay_max=float(h.long_stay_ships.max()),
        daily_mean_min=float(v.groupby("date_local").queue_ships.mean().min()), daily_mean_max=float(v.groupby("date_local").queue_ships.mean().max()),
        kw_time_p=float(kw_time_p),
        **peak_window(j),
    )
    json.dump(kpi, open(C.TABLES / "kpis.json", "w"), indent=2)

    # demurrage scenario table (rates are ASSUMPTIONS from config.py)
    dem = [dict(assumed_rate_usd_per_ship_day=rt, queue_cost_usd=ship_hours / 24 * rt,
                avoided_if_10pct=ship_hours / 24 * rt * 0.10, avoided_if_20pct=ship_hours / 24 * rt * 0.20)
           for rt in C.DEMURRAGE_RATES_USD_DAY]
    pd.DataFrame(dem).round(0).to_csv(C.TABLES / "demurrage_scenarios.csv", index=False)
    decision_table(q1, q2).to_csv(C.TABLES / "decision_table.csv", index=False)

    # ---------------- charts ----------------
    H = C.FINAL_HORIZON
    figs = {
        "V01_zone_map": ch.v01_zone_map(d), "V02_traffic_density": ch.v02_traffic_density(d),
        "V03_hotspots": ch.v03_hotspots(d), "V04_queue_timeline": ch.v04_queue_timeline(d, q1, q2),
        "V05_time_to_berth": ch.v05_journey_distribution(d), "V06_delay_by_vessel": ch.v06_delay_by_vessel(d, fisher_p),
        "V07_arrivals_by_hour": ch.v07_arrivals_by_hour(d), "V08_queue_by_hour": ch.v08_queue_by_hour(d, kw_time_p),
        "V09_correlations": ch.v09_spearman(d), "V10_daily_queue": ch.v10_daily_queue(d, q2),
        "V11_demurrage": ch.v11_demurrage(d, ship_hours, days), "V12_model_vs_naive": ch.v12_model_vs_naive(d),
        "V13_forecast_vs_actual": ch.v13_actual_vs_pred(d, q1, q2, H), "V14_high_hours_caught": ch.v14_high_caught(d, H),
    }
    for name, fig in figs.items():
        fig.savefig(C.CHARTS / f"{name}.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    print("Saved", len(figs), "charts to", C.CHARTS)

    # ---------------- 'so what' line per chart (numbers filled from the data) ----------------
    hs = d["hotspots"]; top = hs[hs.is_hotspot].head(2)
    res, cls = d["model_results"], d["classification_results"]
    pers24 = res[(res.horizon_h == H) & (res.model == "Baseline: same as now")].iloc[0]
    fin24 = res[(res.horizon_h == H) & res.model.str.startswith("Final")].iloc[0]
    pers3 = res[(res.horizon_h == 3) & (res.model == "Baseline: same as now")].iloc[0]
    fin3 = res[(res.horizon_h == 3) & res.model.str.startswith("Final")].iloc[0]
    best = cls[(cls.horizon_h == H) & ~cls.model.str.startswith("Baseline") & ~cls.model.str.startswith("Class from")].sort_values("macro_F1", ascending=False).iloc[0]
    base = cls[(cls.horizon_h == H) & cls.model.str.startswith("Baseline")].iloc[0]
    corr = tests[tests.question.str.contains("wave height")]
    w_raw, w_det = corr.iloc[0].effect_size, corr.iloc[1].effect_size
    b_raw = tests[tests.question == "Queue vs berth occupancy (raw)"].iloc[0].effect_size
    wave_line = (f"Rougher sea went with SHORTER queues (rho {w_raw:+.2f}, {w_det:+.2f} after removing the trend), and ships at berth rise as the queue drains (rho {b_raw:+.2f}): "
                 "the opposite of 'bad weather causes delays'. With 2 weeks of data this is association only; weather added little to forecasts.")
    ins = [
        ("V01", "Study area and zones", "Marine operations", f"{(ais.zone=='anchorage').mean()*100:.0f}% of all pings come from ships sitting still at anchorage and {(ais.zone=='berth').mean()*100:.0f}% from berths: the zones capture real port behaviour."),
        ("V02", "Traffic density", "Port authority", "Ships funnel through two approach lanes; anchorage pressure builds just outside them, so approach scheduling is where coordination pays off."),
        ("V03", "Congestion hotspots (DBSCAN)", "Marine operations", f"{top.iloc[0].hotspot} and {top.iloc[1].hotspot} alone hold {top.share_of_anchorage_ship_hours_pct.sum():.0f}% of all anchor ship-hours: monitor these two spots first."),
        ("V04", "Hourly queue timeline", "Port authority", f"The queue ranges from {v.queue_ships.min():.0f} to {kpi['queue_peak']:.0f} ships and sits in High (>{q2:.0f}) for {kpi['high_hours_pct']:.0f}% of hours: congestion is recurring, not a one-off."),
        ("V05", "Time from port entry to berth", "Shipping company", f"{len(ok)} of {len(r)} ships reach berth in about {kpi['typical_direct_trip_h']:.1f} h; {len(dl)} wait a median of {kpi['median_hours_to_berth_delayed']:.0f} h (up to {kpi['max_hours_to_berth']:.0f} h). Those {kpi['delayed_arrivals']} ships cause {kpi['delayed_share_of_excess_hours_pct']:.0f}% of all delay hours."),
        ("V06", "Delay by vessel type", "Shipping company", f"{tan:.0f}% of tankers are delayed vs {car:.0f}% of cargo ships (Fisher p = {fisher_p:.3f}): tanker berth requests need earlier confirmation."),
        ("V07", "Arrivals by hour", "Marine operations", f"{kpi['arrival_peak_share_pct']:.0f}% of arrivals come between {kpi['arrival_peak_start_local']:02d}:00 and {kpi['arrival_peak_end_local']:02d}:00 local time: roster pilots and tugs for that window."),
        ("V08", "Queue by time of day", "Port authority", f"Queue size does not depend on time of day (Kruskal-Wallis p = {kw_time_p:.2f}): a fixed daily timetable will not fix congestion."),
        ("V09", "Correlations", "Port authority", wave_line),
        ("V10", "Daily average queue", "Logistics planner", f"The daily average queue swings from {kpi['daily_mean_min']:.1f} to {kpi['daily_mean_max']:.1f} ships: plan buffers by the day, not by the hour."),
        ("V11", "Demurrage scenario", "Shipping company", f"At the middle assumed rate (${C.DEMURRAGE_RATES_USD_DAY[1]:,}/ship/day) the queue cost about ${ship_hours/24*C.DEMURRAGE_RATES_USD_DAY[1]/1e6:.1f}M in {days:.0f} days; cutting it 10% avoids about ${ship_hours/24*C.DEMURRAGE_RATES_USD_DAY[1]*0.1/1e6:.2f}M. Rates are assumptions."),
        ("V12", "Model vs naive by horizon", "Port authority", f"At 3 h the naive rule wins (error {pers3.MAE:.2f} vs {fin3.MAE:.2f}); at {H} h the model cuts error by {fin24.skill_vs_persistence_pct:.0f}% ({pers24.MAE:.2f} to {fin24.MAE:.2f} ships). Use the model for day-ahead planning."),
        ("V13", f"{H}-hour forecast vs actual", "Logistics planner", f"Forecasts made a day ahead follow the swings but miss sharp peaks (RMSE {fin24.RMSE:.1f} ships): treat them as a planning signal, not a promise."),
        ("V14", "High-congestion warning", "Marine operations", f"{best.model} catches {int(best.high_caught)} of {int(best.n_high_hours)} High hours with {int(best.false_alarms)} false alarms; the naive rule catches {int(base.high_caught)} with {int(base.false_alarms)} false alarms."),
    ]
    pd.DataFrame(ins, columns=["id", "visual", "client", "so_what"]).to_csv(C.TABLES / "visual_insights.csv", index=False)
    for i in ins: print(f"{i[0]} [{i[2]}] {i[3]}")


if __name__ == "__main__":
    main()
