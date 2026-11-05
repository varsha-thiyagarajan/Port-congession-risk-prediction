"""STEP 5 - Writes report/PROJECT_REPORT.md and report/PRESENTATION_OUTLINE.md.
Every number is read from outputs/tables, so the text can never disagree with the results."""
import json
import pandas as pd
import config as C
from decisions import decision_table

R = C.ROOT / "report"; R.mkdir(exist_ok=True)
k = json.loads((C.TABLES / "kpis.json").read_text())
res = pd.read_csv(C.TABLES / "model_results.csv"); cls = pd.read_csv(C.TABLES / "classification_results.csv")
abl = pd.read_csv(C.TABLES / "feature_ablation.csv"); tests = pd.read_csv(C.TABLES / "stats_tests.csv")
dq = pd.read_csv(C.TABLES / "data_quality.csv"); ins = pd.read_csv(C.TABLES / "visual_insights.csv")
hs = pd.read_csv(C.PROCESSED / "hotspots.csv"); dem = pd.read_csv(C.TABLES / "demurrage_scenarios.csv")
H = C.FINAL_HORIZON
q1, q2 = k["q1"], k["q2"]
span = (pd.Timestamp(k["period_end"]).normalize() - pd.Timestamp(k["period_start"]).normalize()).days + 1


def r(h, name): return res[(res.horizon_h == h) & res.model.str.contains(name, regex=False)].iloc[0]
def md(df): return df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False)


try:
    import tabulate  # noqa
except ImportError:
    def md(df):
        cols = list(df.columns)
        lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
        return "\n".join(lines)

horizon_tbl = pd.DataFrame([{"Horizon": f"{h} h", "Naive error (ships)": r(h, "same as now").MAE, "Model error (ships)": r(h, "Final").MAE,
                             "Model vs naive": f"{r(h, 'Final').skill_vs_persistence_pct:+.0f}%", "Model R2": r(h, "Final").R2} for h in C.HORIZONS]).round(2)
c24 = cls[(cls.horizon_h == H)][["model", "high_caught", "n_high_hours", "false_alarms", "recall_High", "precision_High", "macro_F1"]]
sig = tests[tests.p_value < 0.05][["question", "p_value", "effect_size"]]
top = hs[hs.is_hotspot].head(3)

report = f"""# Coastal Vessel Delay and Port Congestion Risk: Los Angeles / Long Beach

*An AIS-based early-warning system for anchorage queues.*

## 1. Problem
Ships wait at anchor when berths are busy. That costs demurrage and fuel, crowds the anchorage and makes planning hard. Ports usually react after the queue has formed. This project measures the delay, finds what is (and is not) behind it, and forecasts the queue a day ahead so each client can act early.

## 2. Data
| Dataset | Use | Status |
|---|---|---|
| NOAA/USCG AIS, {k['period_start'][:10]} to {k['period_end'][:10]} ({span} days), cargo and tanker ships near LA/LB | Vessel tracks, waiting time, queue | Used |
| Open-Meteo marine and wind (hourly) | Weather features and diagnostics | Used |
| IMF PortWatch | Port-level trend | Script provided, data file not supplied, **not used** |
| AISStream live feed | Live demo | Not built; dashboard replays recorded data |

## 3. Preprocessing
{md(dq.rename(columns={'step': 'Step', 'rows_removed': 'Rows removed', 'rows_remaining': 'Rows remaining', 'pct_of_start': '% of start'}))}

The AIS file was already filtered to the study box, so little needed removing. Zones: berth and anchorage rectangles, a ship counts as *stationary* below {C.STATIONARY_KN:g} kn.
**Target definitions.** *Queue* = ships stationary at anchorage for less than {C.LONG_STAY_H} h (longer stays are lay-ups). *Time to berth* = first berth ping minus first ping inside the port area. *Delayed arrival* = more than {C.DELAY_THRESHOLD_H:g} h to berth.

## 4. Key findings
1. **Most ships glide in, a few get stuck.** {k['arrivals_reached_berth'] - k['delayed_arrivals']} of {k['arrivals_reached_berth']} arrivals reach berth in about {k['typical_direct_trip_h']:.1f} h. {k['delayed_arrivals']} ({k['delayed_share_pct']:.0f}%) take over {C.DELAY_THRESHOLD_H:g} h, a median of {k['median_hours_to_berth_delayed']:.0f} h and up to {k['max_hours_to_berth']:.0f} h. They cause {k['delayed_share_of_excess_hours_pct']:.0f}% of all delay hours.
2. **Tankers are delayed more often**: {k['tanker_delayed_pct']:.0f}% vs {k['cargo_delayed_pct']:.0f}% for cargo ships (Fisher exact p = {k['fisher_p']:.3f}).
3. **Hotspots.** {', '.join(f"{x.hotspot} ({x.ships} ships, {x.share_of_anchorage_ship_hours_pct:.0f}% of anchor hours)" for x in top.itertuples())}.
4. **Congestion is a multi-day swing.** The daily average queue ranges from {k['daily_mean_min']:.1f} to {k['daily_mean_max']:.1f} ships; the queue is High (> {q2:.0f}) for {k['high_hours_pct']:.0f}% of hours. Time of day has no significant effect (Kruskal-Wallis p = {k['kw_time_p']:.2f}).
5. **Weather is not the culprit here.** Wave height is negatively associated with the queue, the opposite of "bad weather causes delay", and it added little to forecasts.
6. **Cost.** The queue represents {k['queue_ship_hours']:,.0f} ship-hours over {k['days']:.1f} days. At an assumed ${C.DEMURRAGE_RATES_USD_DAY[1]:,} per ship per day that is about ${dem.iloc[1].queue_cost_usd / 1e6:.1f}M; a 10% cut would avoid about ${dem.iloc[1].avoided_if_10pct / 1e6:.2f}M. The rate is an assumption.

Significant statistical results:

{md(sig.round(4))}

## 5. Machine learning
**Task.** Forecast the number of ships in the queue H hours ahead, and the Low / Medium / High level (Low ≤ {q1:.0f}, Medium {q1 + 1:.0f}-{q2:.0f}, High > {q2:.0f} ships; terciles of training hours).
**Models.** Naive baselines; Linear Regression (Ridge); Random Forest; LightGBM; final = equal-weight blend of the three, each predicting the *change* from the current queue. Classifiers: LightGBM, Random Forest, Logistic Regression.
**Validation.** Rolling origin: train up to day D, predict day D+1, repeat (strictly time-ordered).

{md(horizon_tbl)}

**Result in one sentence:** the queue changes slowly, so "same as now" is hard to beat for the next few hours; the model adds real value at **{H} hours ahead**, cutting average error by {r(H, 'Final').skill_vs_persistence_pct:.0f}% ({r(H, 'same as now').MAE:.2f} to {r(H, 'Final').MAE:.2f} ships).

High-congestion early warning, {H} h ahead:

{md(c24.round(2))}

**What the model uses.** Only the queue's own recent history and the forecast hour (SHAP: `outputs/charts/shap_summary.png`). SHAP shows mostly mean reversion: when the recent 24 h average queue is high, the model expects it to fall back toward typical levels. Berth counts, arrivals, weekday and weather were tested and did not clear the rule "keep the simplest input set unless another cuts error by more than 10%".
{md(abl[abl.horizon_h == H][['feature_set', 'MAE_blend']].rename(columns={'feature_set': 'Inputs', 'MAE_blend': f'Model error at {H} h'}))}

**Honest caveats.** R2 at {H} h is {r(H, 'Final').R2:.2f}: the model beats the naive rule but explains little variance; it catches the direction of swings and misses sharp peaks. Feature sets were chosen on the same validation days, so the reported error is slightly optimistic. Weak or negative results at 3-12 h are reported, not hidden.

## 6. Decisions by client
{md(decision_table(q1, q2).rename(columns={'congestion': 'Level', 'trigger': 'Trigger', 'client': 'Client', 'action': 'Action'}))}

## 7. Limitations
- {span} days, one port: weekday and seasonal effects cannot be tested; results may not generalise.
- Waiting time is estimated from geofences, not port records. Berth capacity is unknown, so occupancy is a count.
- AIS gaps: {k['low_coverage_hours']} low-coverage hours excluded; some ships go silent for hours.
- Only {k['delayed_arrivals']} delayed arrivals: vessel-type and size comparisons are indicative.
- Cost depends on an assumed demurrage rate and counts ships that may be idle for other reasons.
- Not done: PortWatch analysis, live AISStream feed, hyper-parameter tuning, Isolation Forest (dropped per the scope-control list), cargo handling time.

## 8. Visuals and their "so what"
{md(ins[['id', 'visual', 'client', 'so_what']])}
"""
(R / "PROJECT_REPORT.md").write_text(
    report,
    encoding="utf-8"
)

slides = f"""# Presentation outline (10-12 minutes)

Use the PNGs in `outputs/charts/`. Demo the dashboard on slides 9 and 11.

| # | Slide | Visual | Message |
|---|---|---|---|
| 1 | Title | Dashboard screenshot | Early warning for anchorage queues at LA / Long Beach |
| 2 | Problem | none | Ships wait for berths: demurrage, fuel, crowded anchorage. Ports react late. |
| 3 | Data and cleaning | Data-quality table | AIS ({span} days, {k['ships']} ships) + weather; what was removed and why |
| 4 | Where ships wait | V01, V03 | {ins[ins.id == 'V03'].so_what.iloc[0]} |
| 5 | How long | V05 | {ins[ins.id == 'V05'].so_what.iloc[0]} |
| 6 | Who gets stuck | V06 | {ins[ins.id == 'V06'].so_what.iloc[0]} |
| 7 | What drives the queue | V08, V10, V09 | Not time of day, not weather: multi-day swings. |
| 8 | Cost | V11 | {ins[ins.id == 'V11'].so_what.iloc[0]} |
| 9 | Forecast (live demo) | Dashboard > Forecast | Day-ahead queue and level with chance of High |
| 10 | Does ML help? | V12, V13, V14 | {ins[ins.id == 'V12'].so_what.iloc[0]} |
| 11 | What drives the model | shap_summary.png | The queue's own recent history dominates |
| 12 | Decisions | Dashboard > Cost & decisions | Action per client at Low / Medium / High |
| 13 | Limitations and next steps | none | More months, more ports, berth capacity data, live AIS feed |

## Speaker tip
Lead with the honest ML message: "we tested it against the simplest possible rule, and it only wins at a day ahead." It makes the rest of the project more credible.
"""
(R / "PRESENTATION_OUTLINE.md").write_text(slides)
print("Wrote report/PROJECT_REPORT.md and report/PRESENTATION_OUTLINE.md")
