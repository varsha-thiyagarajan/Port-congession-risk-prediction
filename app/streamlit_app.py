"""
Port Congestion Early-Warning Dashboard  -  Los Angeles / Long Beach
Run:  streamlit run app/streamlit_app.py
Everything on screen is read from data/processed and outputs/ (created by run_pipeline.py). Nothing is typed in.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config as C
from common import congestion_class, to_local
from data_access import load_all, load_models, valid_hours
from decisions import CLIENTS, decision_table
from features import FEATURE_LABELS, FEATURES, build

st.set_page_config(page_title="LA/LB Port Congestion Early Warning", page_icon="⚓", layout="wide")

NAVY, TEAL, ORANGE, AMBER, GREY, RED, GREEN = "#1f3a5f", "#2a9d8f", "#e76f51", "#e9c46a", "#8d99ae", "#d62828", "#2a9d8f"
LEVEL_COL = {"Low": GREEN, "Medium": AMBER, "High": ORANGE}

st.markdown(f"""
<style>
.block-container {{padding-top: 1.6rem;}}
.kpi {{background:#f6f8fb;border-left:5px solid {NAVY};padding:12px 16px;border-radius:6px;}}
.kpi .v {{font-size:1.9rem;font-weight:700;color:{NAVY};line-height:1.1}}
.kpi .l {{font-size:.82rem;color:#5b6b7f;margin-top:2px}}
.badge {{display:inline-block;padding:4px 14px;border-radius:14px;color:white;font-weight:700;font-size:1.05rem}}
.sowhat {{background:#eef7f5;border-left:4px solid {TEAL};padding:8px 12px;border-radius:4px;margin:4px 0 14px 0;font-size:.95rem}}
.warn {{background:#fff6ee;border-left:4px solid {ORANGE};padding:8px 12px;border-radius:4px;margin:6px 0;font-size:.92rem}}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading port data…")
def get_data():
    d = load_all()
    d["ais"] = d["ais"].sort_values("base_date_time").reset_index(drop=True)
    d["valid"] = valid_hours(d["hourly"])
    d["kpi"] = json.loads((C.TABLES / "kpis.json").read_text())
    d["insights"] = pd.read_csv(C.TABLES / "visual_insights.csv")
    d["demurrage"] = pd.read_csv(C.TABLES / "demurrage_scenarios.csv")
    return d


@st.cache_resource
def get_models():
    models = load_models()

    for mdl in models.values():
        classifier = mdl.get("classifier")

        if classifier is None:
            continue

        # Handle Pipeline and standalone classifiers
        if hasattr(classifier, "steps"):
            final_classifier = classifier.steps[-1][1]
        else:
            final_classifier = classifier

        if (
            final_classifier.__class__.__name__ == "LogisticRegression"
            and not hasattr(final_classifier, "multi_class")
        ):
            final_classifier.multi_class = "auto"

    return models


needed = [C.PROCESSED / "hourly_table.csv", C.TABLES / "kpis.json", C.MODELS / f"queue_forecast_{C.FINAL_HORIZON}h.joblib"]
if not all(p.exists() for p in needed):
    st.error("Processed data not found. Run `python run_pipeline.py` from the project folder first.")
    st.stop()

D = get_data()
M = get_models()
H_, J_, V_, K_ = D["hourly"], D["journeys"], D["valid"], D["kpi"]
Q1, Q2 = M[C.FINAL_HORIZON]["q1"], M[C.FINAL_HORIZON]["q2"]
SPAN_DAYS = (pd.Timestamp(K_["period_end"]).normalize() - pd.Timestamp(K_["period_start"]).normalize()).days + 1


def local_to_utc(ts):
    return pd.Timestamp(ts).tz_localize(C.TZ_LOCAL).tz_convert("UTC").tz_localize(None)


def level_of(q):
    return str(congestion_class([q], Q1, Q2)[0])


def badge(level, text=None):
    return f'<span class="badge" style="background:{LEVEL_COL[level]}">{text or level}</span>'


def kpi(label, value):
    return f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>'


def sowhat(text):
    st.markdown(f'<div class="sowhat"><b>So what:</b> {text}</div>', unsafe_allow_html=True)


def chart(name, caption=None):
    f = C.CHARTS / f"{name}.png"
    if f.exists():
        st.image(str(f), width="stretch")
    if caption:
        sowhat(caption)


def insight(vid):
    r = D["insights"][D["insights"].id == vid].iloc[0]
    return r.so_what, r.client


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("⚓ LA / Long Beach")
st.sidebar.caption("AIS-based early warning for vessel waiting and port congestion")
PAGES = ["🏠 Overview", "🗺️ Port map (replay)", "🔮 Forecast", "🚨 Alerts", "📊 Insights gallery", "💰 Cost & decisions", "ℹ️ Data & method"]
page = st.sidebar.radio("Go to", PAGES, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("**Replay clock** (local Pacific time)")
local_idx = to_local(V_.index)
t_local = st.sidebar.select_slider("Time", options=list(local_idx), value=local_idx[len(local_idx) // 2],
                                   format_func=lambda x: x.strftime("%a %b %d, %H:%M"), label_visibility="collapsed")
T_UTC = local_to_utc(t_local)
q_at_t = V_.queue_ships.get(T_UTC, np.nan)
lvl_t = level_of(q_at_t) if pd.notna(q_at_t) else "Low"
st.sidebar.markdown(f"Queue at this hour: **{q_at_t:.0f} ships** &nbsp; {badge(lvl_t)}", unsafe_allow_html=True)
st.sidebar.markdown("---")
RATE = st.sidebar.number_input("Assumed demurrage (USD / ship / day)", min_value=1000, max_value=200000,
                               value=int(C.DEMURRAGE_RATES_USD_DAY[1]), step=1000,
                               help="An ASSUMPTION used for cost estimates. Replace it with a rate from your charter parties.")
st.sidebar.caption(f"Data: NOAA/USCG AIS, {K_['period_start'][:10]} to {K_['period_end'][:10]}; Open-Meteo weather.")


# ----------------------------------------------------------------------------- helpers for ship state
def ships_state(t_utc, window_min=30):
    ais = D["ais"]
    lo = ais.base_date_time.searchsorted(t_utc - pd.Timedelta(minutes=window_min), side="right")
    hi = ais.base_date_time.searchsorted(t_utc, side="right")
    w = ais.iloc[lo:hi]
    s = w.groupby("mmsi").tail(1).copy()
    eps = D["episodes"]
    waited = {}
    for r in s[s.zone == "anchorage"].itertuples():
        e = eps[(eps.mmsi == r.mmsi) & (eps.start <= t_utc) & (eps.end >= t_utc - pd.Timedelta(hours=1))]
        if len(e):
            waited[r.mmsi] = (t_utc - e.iloc[0].start).total_seconds() / 3600
    s["waited_h"] = s.mmsi.map(waited)
    s["state"] = s.zone.map({"anchorage": "At anchor", "berth": "At berth", "other": "Moving / approaching"})
    return s


def alert_level(hours):
    if hours >= C.LONG_STAY_H:
        return "Critical (long stay)"
    if hours >= K_["median_hours_to_berth_delayed"]:
        return "Alert"
    if hours >= C.DELAY_THRESHOLD_H:
        return "Watch"
    return "Normal"


# ============================================================================= PAGE: OVERVIEW
if page == PAGES[0]:
    st.title("Port congestion early warning: Los Angeles / Long Beach")
    st.markdown("**From raw ship positions to a day-ahead congestion forecast.** "
                f"{K_['ships']} cargo ships and tankers tracked by AIS for {SPAN_DAYS} days (October 2025).")
    c = st.columns(5)
    c[0].markdown(kpi("Ship arrivals observed", f"{K_['arrivals_reached_berth']}"), unsafe_allow_html=True)
    c[1].markdown(kpi(f"Delayed more than {C.DELAY_THRESHOLD_H:g} h", f"{K_['delayed_arrivals']} ({K_['delayed_share_pct']:.0f}%)"), unsafe_allow_html=True)
    c[2].markdown(kpi("Of all delay hours caused by those ships", f"{K_['delayed_share_of_excess_hours_pct']:.0f}%"), unsafe_allow_html=True)
    c[3].markdown(kpi("Peak queue (ships at anchor)", f"{K_['queue_peak']:.0f}"), unsafe_allow_html=True)
    c[4].markdown(kpi(f"Hours in High congestion (> {Q2:.0f} ships)", f"{K_['high_hours_pct']:.0f}%"), unsafe_allow_html=True)

    st.markdown("### The story in five steps")
    st.markdown("#### 1. Where do ships wait?")
    a, b = st.columns(2)
    with a:
        chart("V01_zone_map")
    with b:
        chart("V03_hotspots")
    sowhat(insight("V03")[0])

    st.markdown("#### 2. How long do they wait, and who gets stuck?")
    a, b = st.columns([3, 2])
    with a:
        chart("V05_time_to_berth")
    with b:
        chart("V06_delay_by_vessel")
    sowhat(insight("V05")[0] + " " + insight("V06")[0])

    st.markdown("#### 3. What drives the queue?")
    a, b = st.columns(2)
    with a:
        chart("V10_daily_queue")
    with b:
        chart("V08_queue_by_hour")
    sowhat(insight("V10")[0] + " " + insight("V08")[0])

    st.markdown("#### 4. Can we see it coming?")
    a, b = st.columns(2)
    with a:
        chart("V12_model_vs_naive")
    with b:
        chart("V13_forecast_vs_actual")
    sowhat(insight("V12")[0])

    st.markdown("#### 5. What should each client do?")
    st.markdown("Open **💰 Cost & decisions** for the action table and the demurrage calculator, "
                "**🔮 Forecast** for the live prediction, and **🚨 Alerts** for ships that need attention now.")


# ============================================================================= PAGE: MAP
elif page == PAGES[1]:
    st.title("Port map: replay any hour")
    st.caption("Move the replay clock in the sidebar. Ships are the last AIS position in the 30 minutes before the selected time.")
    s = ships_state(T_UTC)
    anch, berth, mov = (s.state == "At anchor"), (s.state == "At berth"), (s.state == "Moving / approaching")
    c = st.columns(4)
    c[0].metric("At anchor (all)", int(anch.sum()))
    c[1].metric("Queue (anchor < 3 days)", f"{q_at_t:.0f}")
    c[2].metric("At berth", int(berth.sum()))
    c[3].markdown(f"**Congestion**<br>{badge(lvl_t)}", unsafe_allow_html=True)
    o1, o2, o3 = st.columns(3)
    tiles = o1.checkbox("Street map background (needs internet)", value=True)
    show_hot = o2.checkbox("Show congestion hotspots (DBSCAN)", value=True)
    show_zone = o3.checkbox("Show geofence zones", value=True)

    COL = {"At anchor": RED, "At berth": NAVY, "Moving / approaching": GREY}
    fig = go.Figure()
    Trace = go.Scattermap if tiles else go.Scatter

    def txt(r):
        base = f"MMSI {r.mmsi}<br>{r.vessel_group}, {r.length:.0f} m<br>{r.sog:.1f} kn"
        return base + (f"<br><b>waiting {r.waited_h:.0f} h</b>" if pd.notna(r.waited_h) else "")

    if show_zone:
        for box, col, nm in [(C.BERTH, NAVY, "Berth zone"), (C.BERTH_W, NAVY, "Berth zone"), (C.ANCH_IN, RED, "Anchorage zone"), (C.ANCH_OUT, RED, "Anchorage zone")]:
            xs, ys = [box[0], box[1], box[1], box[0], box[0]], [box[2], box[2], box[3], box[3], box[2]]
            kw = dict(lon=xs, lat=ys) if tiles else dict(x=xs, y=ys)
            fig.add_trace(Trace(mode="lines", line=dict(color=col, width=2), hoverinfo="skip", showlegend=False, **kw))
    if show_hot:
        hs = D["hotspots"].query("is_hotspot")
        kw = dict(lon=hs.lon, lat=hs.lat) if tiles else dict(x=hs.lon, y=hs.lat)
        mk = dict(size=np.clip(hs.ships * 3 + 18, 20, 60), color="rgba(231,111,81,0.25)") if tiles else dict(size=np.clip(hs.ships * 3 + 18, 20, 60), color="rgba(231,111,81,0.25)")
        fig.add_trace(Trace(mode="markers", marker=mk, text=[f"{r.hotspot}: {r.ships} ships, {r.share_of_anchorage_ship_hours_pct:.0f}% of anchor hours" for r in hs.itertuples()],
                            hoverinfo="text", name="Hotspots", **kw))
    for state in ["Moving / approaching", "At anchor", "At berth"]:
        g = s[s.state == state]
        kw = dict(lon=g.longitude, lat=g.latitude) if tiles else dict(x=g.longitude, y=g.latitude)
        fig.add_trace(Trace(mode="markers", marker=dict(size=9, color=COL[state]), name=f"{state} ({len(g)})",
                            text=[txt(r) for r in g.itertuples()], hoverinfo="text", **kw))
    if tiles:
        fig.update_layout(map=dict(style="carto-positron", center=dict(lat=33.69, lon=-118.17), zoom=10.2))
    else:
        fig.update_xaxes(range=[-118.5, -117.95], title="Longitude")
        fig.update_yaxes(range=[33.55, 33.82], title="Latitude", scaleanchor="x", scaleratio=1 / np.cos(np.radians(33.7)))
    fig.update_layout(height=640, margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=1.02, x=0))
    st.plotly_chart(fig, width="stretch")
    sowhat("Red dots are ships sitting at anchor. Their number, not the time of day, is what the early-warning model tracks. "
           "Orange halos mark the places where ships pile up most.")

    st.markdown("##### Ships waiting at anchor right now")
    t = s[anch].sort_values("waited_h", ascending=False)[["mmsi", "vessel_group", "length", "waited_h"]].rename(
        columns={"mmsi": "MMSI", "vessel_group": "Type", "length": "Length (m)", "waited_h": "Waiting so far (h)"})
    st.dataframe(t, hide_index=True, width="stretch",
                 column_config={"Waiting so far (h)": st.column_config.NumberColumn(format="%.1f"), "Length (m)": st.column_config.NumberColumn(format="%.0f")})


# ============================================================================= PAGE: FORECAST
elif page == PAGES[2]:
    st.title("Forecast: how many ships will be waiting?")
    res, cls, abl, oof = D["model_results"], D["classification_results"], D["feature_ablation"], D["predictions_oof"]
    Hsel = st.radio("Forecast horizon", C.HORIZONS, index=C.HORIZONS.index(C.FINAL_HORIZON), horizontal=True, format_func=lambda x: f"{x} hours ahead")
    mdl = M[Hsel]
    fin = res[(res.horizon_h == Hsel) & res.model.str.startswith("Final")].iloc[0]
    per = res[(res.horizon_h == Hsel) & (res.model == "Baseline: same as now")].iloc[0]
    useful = fin.MAE < per.MAE
    if useful:
        st.success(f"At {Hsel} h the model is {fin.skill_vs_persistence_pct:.0f}% more accurate than assuming the queue stays as it is "
                   f"(average error {fin.MAE:.2f} vs {per.MAE:.2f} ships on {int(fin.n_test_hours)} unseen hours).")
    else:
        st.markdown(f'<div class="warn"><b>Honest result:</b> at {Hsel} h the simple rule “the queue stays as it is now” '
                    f"(error {per.MAE:.2f} ships) is as good as or better than the model ({fin.MAE:.2f}). "
                    f"The model only adds value at longer horizons, so use the current queue for decisions this soon.</div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Latest forecast", "Replay: forecast vs what happened", "What-if", "Model card"])

    def predict(row_df, mdl):
        X = row_df[FEATURES]
        delta = (mdl["ridge"].predict(X) + mdl["rf"].predict(X) + mdl["lgbm"].predict(X)) / 3
        q = row_df.queue_now.values + delta
        proba = pd.Series(mdl["classifier"].predict_proba(X)[0], index=mdl["classifier"].classes_)
        return float(q[0]), proba

    with tab1:
        f = build(H_, Hsel, with_target=False)
        origin = f.index[-1]
        q_hat, proba = predict(f.iloc[[-1]], mdl)
        target_local = to_local([origin + pd.Timedelta(hours=Hsel)])[0]
        lvl = level_of(q_hat)
        st.markdown(f"Forecast made from the last full hour of data (**{to_local([origin])[0]:%a %b %d, %H:%M}** local) "
                    f"for **{target_local:%a %b %d, %H:%M}** local.")
        c = st.columns(4)
        c[0].metric("Queue now", f"{f.iloc[-1].queue_now:.0f} ships")
        c[1].metric(f"Forecast in {Hsel} h", f"{q_hat:.1f} ships")
        c[2].markdown(f"**Expected congestion**<br>{badge(lvl)}", unsafe_allow_html=True)
        c[3].metric("Chance of High", f"{proba.get('High', 0) * 100:.0f}%", help=f"From the {mdl['classifier_name']}. A risk score, not a calibrated probability.")
        pr = pd.DataFrame({"Level": ["Low", "Medium", "High"], "Probability": [proba.get(k, 0) for k in ["Low", "Medium", "High"]]})
        figp = go.Figure(go.Bar(x=pr.Probability * 100, y=pr.Level, orientation="h", marker_color=[LEVEL_COL[k] for k in pr.Level],
                                text=[f"{v * 100:.0f}%" for v in pr.Probability], textposition="outside"))
        figp.update_layout(height=190, margin=dict(l=0, r=30, t=10, b=0), xaxis=dict(range=[0, 110], title="Probability (%)"), showlegend=False)
        st.plotly_chart(figp, width="stretch")
        st.markdown(f"**Level definitions** (set from training data): Low ≤ {Q1:.0f} ships, Medium {Q1 + 1:.0f}–{Q2:.0f}, High > {Q2:.0f}.")
        st.markdown("**What to do now:**")
        dt = decision_table(Q1, Q2)
        st.dataframe(dt[dt.congestion == lvl][["client", "action"]].rename(columns={"client": "Client", "action": "Action"}), hide_index=True, width="stretch")

    with tab2:
        st.caption("Each forecast below was made using only data available at the time, by a model trained on earlier days. Nothing leaks from the future.")
        o = oof[oof.horizon_h == Hsel].copy()
        o["origin_utc"] = pd.to_datetime(o.origin_utc)
        o["target_local"] = to_local(o.origin_utc + pd.Timedelta(hours=Hsel))
        o["origin_local"] = to_local(o.origin_utc)
        o = o.sort_values("origin_local")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=o.target_local, y=o.actual, name="Actual queue", line=dict(color="black", width=3)))
        fig.add_trace(go.Scatter(x=o.target_local, y=o.persistence, name="Naive: queue stays as now", line=dict(color=GREY, width=1.5)))
        fig.add_trace(go.Scatter(x=o.target_local, y=o.blend, name="Model forecast", line=dict(color=TEAL, width=2.5, dash="dash")))
        fig.add_hline(y=Q2, line_dash="dot", line_color=ORANGE, annotation_text=f"High above {Q2:.0f}")
        sel_origin = t_local
        near = o.iloc[(o.origin_local - sel_origin).abs().argsort()[:1]]
        if len(near) and abs((near.origin_local.iloc[0] - sel_origin).total_seconds()) <= 3600:
            r = near.iloc[0]
            fig.add_trace(go.Scatter(x=[r.target_local], y=[r.blend], mode="markers", marker=dict(size=14, color=ORANGE, symbol="diamond"), name="Replay clock forecast"))
            st.info(f"At the replay clock ({sel_origin:%a %H:%M}) the model forecast **{r.blend:.1f} ships** for {r.target_local:%a %H:%M}. "
                    f"What actually happened: **{r.actual:.0f}** ships. The naive rule said {r.persistence:.0f}.")
        else:
            st.caption("Move the replay clock into the shaded test period to see a single forecast highlighted.")
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Ships in queue", xaxis_title="Local time (hour being forecast)", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")
        a, b = st.columns(2)
        with a:
            chart("V14_high_hours_caught")
        with b:
            c14 = cls[(cls.horizon_h == Hsel) & ~cls.model.str.startswith("Class from")]
            st.markdown("**High-congestion early warning, same unseen hours**")
            st.dataframe(c14[["model", "high_caught", "n_high_hours", "false_alarms", "recall_High", "precision_High", "macro_F1"]]
                         .rename(columns={"model": "Method", "high_caught": "High hours caught", "n_high_hours": "High hours in test", "false_alarms": "False alarms",
                                          "recall_High": "Recall (High)", "precision_High": "Precision (High)", "macro_F1": "Macro F1"}),
                         hide_index=True, width="stretch")

    with tab3:
        st.caption("Type in a situation and see what the model expects. Defaults are the latest hour of real data.")
        f = build(H_, Hsel, with_target=False)
        last = f.iloc[-1]
        a, b, c = st.columns(3)
        qn = a.number_input("Ships queueing now", 0, 40, int(last.queue_now))
        q3 = b.number_input("…3 hours ago", 0, 40, int(last.queue_3h_ago))
        q6 = c.number_input("…6 hours ago", 0, 40, int(last.queue_6h_ago))
        a, b, c = st.columns(3)
        qm = a.number_input("Average queue, last 24 h", 0.0, 40.0, float(round(last.queue_mean_24h, 1)), 0.5)
        qs = b.number_input("Queue variability (std), last 24 h", 0.0, 15.0, float(round(last.queue_std_24h, 1)), 0.1)
        hr = c.slider("Local hour being forecast", 0, 23, int(to_local([f.index[-1] + pd.Timedelta(hours=Hsel)])[0].hour))
        row = pd.DataFrame([dict(queue_now=qn, queue_3h_ago=q3, queue_6h_ago=q6, queue_mean_24h=qm, queue_std_24h=qs, queue_trend_3h=qn - q3,
                                 target_hour_sin=np.sin(2 * np.pi * hr / 24), target_hour_cos=np.cos(2 * np.pi * hr / 24))])
        q_hat, proba = predict(row, mdl)
        lv = level_of(q_hat)
        c = st.columns(3)
        c[0].metric(f"Forecast in {Hsel} h", f"{q_hat:.1f} ships")
        c[1].markdown(f"**Expected congestion**<br>{badge(lv)}", unsafe_allow_html=True)
        c[2].metric("Chance of High", f"{proba.get('High', 0) * 100:.0f}%")

    with tab4:
        st.markdown("##### How the model was tested")
        st.markdown(f"Rolling-origin validation: train on all days up to day *D*, forecast day *D+1*, repeat. "
                    f"Final models are trained on all {mdl['trained_on_hours']} labelled hours. "
                    f"Predictions are the **change** from the current queue, averaged over three models (Ridge, Random Forest, LightGBM).")
        r = res[res.horizon_h == Hsel][["model", "MAE", "RMSE", "R2", "skill_vs_persistence_pct"]].rename(
            columns={"model": "Model", "skill_vs_persistence_pct": "Better than naive (%)"})
        st.dataframe(r, hide_index=True, width="stretch")
        st.caption("MAE/RMSE are in ships. R² can be negative on a short test period: it means the model is no better than guessing the test period's own average.")
        a, b = st.columns(2)
        with a:
            st.markdown("**What drives the forecast (SHAP)**")
            if (C.CHARTS / "shap_summary.png").exists():
                st.image(str(C.CHARTS / "shap_summary.png"), width="stretch")
            st.caption("SHAP for the 24 h LightGBM model; the model predicts the CHANGE from the current queue. Each dot is one hour; red = high feature value. Reading: when the last-24 h average queue is high, the model expects the queue to fall back (mean reversion). It learned typical swing behaviour, not hidden causes.")
        with b:
            st.markdown("**Did extra information help? (feature test)**")
            ab = abl[abl.horizon_h == Hsel][["feature_set", "MAE_blend", "MAE_persistence"]].rename(columns={"feature_set": "Inputs", "MAE_blend": "Model error", "MAE_persistence": "Naive error"})
            st.dataframe(ab, hide_index=True, width="stretch")
            st.caption("Rule used: keep the simplest input set unless another cuts error by more than 10%. Berth counts, weekday and weather did not clear that bar.")


# ============================================================================= PAGE: ALERTS
elif page == PAGES[3]:
    st.title("Alerts")
    st.caption(f"Status at the replay clock: **{t_local:%a %b %d, %H:%M}** local. Alert levels come from the data: Watch > {C.DELAY_THRESHOLD_H:g} h, "
               f"Alert > {K_['median_hours_to_berth_delayed']:.0f} h (median of delayed ships), Critical ≥ {C.LONG_STAY_H} h.")
    s = ships_state(T_UTC)
    a1, a2, a3 = st.tabs(["Late-arrival alert table", "Long-stay & anomaly alerts", "Queue surge alerts"])

    with a1:
        j = J_[(J_.arrival_time <= T_UTC) & (J_.last_seen >= T_UTC - pd.Timedelta(hours=1)) & (J_.berth_time.isna() | (J_.berth_time > T_UTC))].copy()
        j["hours_since_entry"] = (T_UTC - j.arrival_time).dt.total_seconds() / 3600
        j = j[j.hours_since_entry >= C.DELAY_THRESHOLD_H]
        where = s.set_index("mmsi").state
        j["where_now"] = j.mmsi.map(where).fillna("Out of range")
        j["alert"] = j.hours_since_entry.map(alert_level)
        j["demurrage_so_far"] = (j.hours_since_entry - j.typical_transit_h).clip(lower=0) / 24 * RATE
        st.markdown(f"**{len(j)} ships have been in the port area for more than {C.DELAY_THRESHOLD_H:g} h without reaching a berth.**")
        out = j.sort_values("hours_since_entry", ascending=False)[["mmsi", "vessel_group", "length", "arrival_time", "hours_since_entry", "where_now", "alert", "demurrage_so_far"]].copy()
        out["arrival_time"] = to_local(out.arrival_time).strftime("%a %b %d %H:%M") if len(out) else out.arrival_time
        st.dataframe(out.rename(columns={"mmsi": "MMSI", "vessel_group": "Type", "length": "Length (m)", "arrival_time": "Entered port area (local)",
                                         "hours_since_entry": "Hours since entry", "where_now": "Where now", "alert": "Alert", "demurrage_so_far": f"Est. delay cost so far (USD @ ${RATE:,}/day)"}),
                     hide_index=True, width="stretch",
                     column_config={"Hours since entry": st.column_config.NumberColumn(format="%.1f"), "Length (m)": st.column_config.NumberColumn(format="%.0f"),
                                    f"Est. delay cost so far (USD @ ${RATE:,}/day)": st.column_config.NumberColumn(format="$%.0f")})
        sowhat("These ships entered the port area and have not yet reached a berth. Shipping companies can warn charterers; logistics planners can push back pickups.")

    with a2:
        eps = D["episodes"]
        cur = eps[(eps.start <= T_UTC) & (eps.end >= T_UTC - pd.Timedelta(hours=1))].copy()
        cur["hours"] = (T_UTC - cur.start).dt.total_seconds() / 3600
        long_ = cur[cur.hours >= C.LONG_STAY_H].sort_values("hours", ascending=False)
        st.markdown(f"**Long-stay ships (at anchor ≥ {C.LONG_STAY_H} h): {len(long_)}**")
        st.dataframe(long_[["mmsi", "vessel_group", "length", "hours"]].rename(columns={"mmsi": "MMSI", "vessel_group": "Type", "length": "Length (m)", "hours": "At anchor (h)"}),
                     hide_index=True, width="stretch", column_config={"At anchor (h)": st.column_config.NumberColumn(format="%.0f")})
        st.markdown(f"**Ships seen stationary outside any zone, or with a long AIS silence:**")
        gaps = J_[(J_.max_gap_h > 6)].sort_values("max_gap_h", ascending=False).head(10)
        st.dataframe(gaps[["mmsi", "vessel_group", "max_gap_h", "status"]].rename(columns={"mmsi": "MMSI", "vessel_group": "Type", "max_gap_h": "Longest AIS silence (h)", "status": "Journey outcome"}),
                     hide_index=True, width="stretch", column_config={"Longest AIS silence (h)": st.column_config.NumberColumn(format="%.1f")})
        sowhat("Long-stay ships are excluded from the queue (they are lay-ups, not waiting for a berth) but marine operations should still track them. "
               "AIS silences over 6 h make a waiting-time estimate less reliable.")

    with a3:
        d3 = (V_.queue_ships - V_.queue_ships.shift(3)).dropna()
        thr = max(3.0, float(d3.mean() + 2 * d3.std()))
        surge = d3[d3 >= thr]
        st.markdown(f"A **queue surge** = queue rises by at least **{thr:.0f} ships in 3 hours** (mean + 2 standard deviations of all 3-hour changes, minimum 3).")
        st.markdown(f"Found **{len(surge)}** surge hours in the period.")
        if len(surge):
            tt = pd.DataFrame({"Time (local)": to_local(surge.index).strftime("%a %b %d %H:%M"), "Rise in 3 h (ships)": surge.values,
                               "Queue after": V_.queue_ships.reindex(surge.index).values})
            st.dataframe(tt, hide_index=True, width="stretch")
        sowhat("Surges are the moments when several ships reach anchor together: that is when extra tugs and pilots help most.")


# ============================================================================= PAGE: INSIGHTS
elif page == PAGES[4]:
    st.title("Insights gallery")
    ins = D["insights"]
    cl = st.multiselect("Show visuals for client", sorted(ins.client.unique()), default=list(sorted(ins.client.unique())))
    st.caption("Every visual names the client it is for and a one-line “so what”.")
    sub = ins[ins.client.isin(cl)]
    files = {p.name[:3]: p for p in C.CHARTS.glob("V*.png")}
    for r in sub.itertuples():
        st.markdown(f"#### {r.id} · {r.visual} &nbsp; <span style='font-size:.8rem;color:#5b6b7f'>for {r.client}</span>", unsafe_allow_html=True)
        if r.id in files:
            st.image(str(files[r.id]), width="stretch")
        sowhat(r.so_what)
    st.markdown("### Statistical tests behind the insights")
    st.dataframe(D["stats_tests"][["test", "question", "n", "p_value", "effect_size", "effect_measure", "conclusion"]], hide_index=True, width="stretch")
    st.caption("With only about two weeks of data, treat significant results as leads to verify, not as proof.")


# ============================================================================= PAGE: COST & DECISIONS
elif page == PAGES[5]:
    st.title("Cost and decisions")
    t1, t2 = st.tabs(["Demurrage scenario", "Decision table"])
    with t1:
        st.caption("Cost = waiting ship-hours × assumed daily demurrage rate ÷ 24. The rate is an assumption you set in the sidebar.")
        red = st.slider("Reduction in waiting achieved by acting on the early warning", 0, 50, 10, 5, format="%d%%")
        ship_hours = K_["queue_ship_hours"]
        cost_all = ship_hours / 24 * RATE
        dl = J_[J_.delayed == 1]
        cost_del = dl.excess_delay_h.sum() / 24 * RATE
        c = st.columns(3)
        c[0].markdown(kpi(f"Queue cost, {K_['days']:.0f} days (all ships idle at anchor)", f"${cost_all / 1e6:.2f}M"), unsafe_allow_html=True)
        c[1].markdown(kpi(f"Avoided with a {red}% cut", f"${cost_all * red / 100 / 1e6:.2f}M"), unsafe_allow_html=True)
        c[2].markdown(kpi(f"Cost of the {len(dl)} clearly delayed arrivals only", f"${cost_del / 1e6:.2f}M"), unsafe_allow_html=True)
        st.markdown(f"Based on **{ship_hours:,.0f} ship-hours** in the queue over **{K_['days']:.1f} days** (about {ship_hours / K_['days']:.0f} ship-hours per day). "
                    f"The second figure counts only the extra hours of the {len(dl)} ships that took more than {C.DELAY_THRESHOLD_H:g} h from port entry to berth, so it is the more cautious number.")
        rates = np.array(C.DEMURRAGE_RATES_USD_DAY + ([RATE] if RATE not in C.DEMURRAGE_RATES_USD_DAY else []))
        rates.sort()
        figc = go.Figure()
        figc.add_bar(x=[f"${r:,}" for r in rates], y=ship_hours / 24 * rates / 1e6, name="Queue cost", marker_color=NAVY)
        figc.add_bar(x=[f"${r:,}" for r in rates], y=ship_hours / 24 * rates * red / 100 / 1e6, name=f"Avoided ({red}% cut)", marker_color=TEAL)
        figc.update_layout(barmode="group", height=360, yaxis_title="Million USD", xaxis_title="Assumed demurrage rate (USD per ship per day)", margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(figc, width="stretch")
        st.markdown('<div class="warn">Not every ship at anchor is incurring demurrage (some wait for bunkering or crew change). '
                    "Treat the first number as an upper bound and the second as a lower bound.</div>", unsafe_allow_html=True)
    with t2:
        dt = decision_table(Q1, Q2)
        st.markdown(f"Current replay hour: **{q_at_t:.0f} ships** → {badge(lvl_t)}", unsafe_allow_html=True)
        cl = st.multiselect("Clients", CLIENTS, default=CLIENTS)
        show_all = st.checkbox("Show all three levels", value=False)
        d2 = dt[dt.client.isin(cl)]
        if not show_all:
            d2 = d2[d2.congestion == lvl_t]
        st.dataframe(d2.rename(columns={"congestion": "Level", "trigger": "Trigger", "client": "Client", "action": "Action"}), hide_index=True, width="stretch")
        st.caption("Triggers use the queue size thresholds learned from the training hours (terciles).")


# ============================================================================= PAGE: ABOUT
else:
    st.title("Data and method")
    st.markdown("""
**Pipeline** (all code in `src/`, run by `run_pipeline.py`):
1. *Extract* raw NOAA/USCG AIS files → cargo and tanker ships near LA/LB. 2. *Download* hourly wave and wind data (Open-Meteo).
3. *Clean* (invalid positions, missing or impossible speeds, duplicates, GPS jumps) and *geofence* (berth and anchorage zones; stationary = under 1 knot).
4. *Measure*: queue per hour; for each arriving ship, time from port entry to berth. 5. *Analyse*: descriptive, spatial (DBSCAN), diagnostic (Spearman, Kruskal-Wallis, Fisher), demurrage scenario.
6. *Forecast*: queue size and Low/Medium/High level 3–24 h ahead, tested on unseen days. 7. *Decide*: actions per client.
""")
    st.markdown("##### Data quality: rows before and after cleaning")
    st.dataframe(D["dq"].rename(columns={"step": "Step", "rows_removed": "Rows removed", "rows_remaining": "Rows remaining", "pct_of_start": "% of start"}), hide_index=True, width="stretch")
    st.markdown("##### Limitations (please read before quoting numbers)")
    st.markdown(f"""
- **Short period.** {SPAN_DAYS} days of one port. Patterns such as weekday effects cannot be established; results may not generalise.
- **Waiting time is estimated from geofences**, not official port records. Berth capacity is unknown, so occupancy is a ship count, not a percentage.
- **AIS has gaps** ({K_['low_coverage_hours']} low-coverage hours are excluded from queue statistics; some ships go silent for hours).
- **Delayed ships are few** ({K_['delayed_arrivals']} of {K_['arrivals_reached_berth']}), so vessel-type and size comparisons are indicative.
- **Forecast skill is limited.** Only the 24-hour horizon beats the naive rule. Validation days were also used to choose between feature sets.
- **Costs depend on an assumed demurrage rate** and include ships that may be idle for other reasons.
- **Not included:** IMF PortWatch (file not supplied), live AISStream feed (optional; the dashboard replays recorded data), cargo handling time.
""")
