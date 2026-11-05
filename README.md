# Port Congestion Early Warning: Los Angeles / Long Beach

AIS-based analytics + day-ahead queue forecast + Streamlit dashboard.
About 65% analytics (14 story charts, statistical tests, hotspots, cost scenario) and 35% machine learning (queue forecast, congestion level, SHAP).

## Quick start
```bash
pip install -r requirements.txt
python run_pipeline.py              # rebuilds every table, model, chart and the report (about 1 minute)
streamlit run app/streamlit_app.py  # opens the dashboard
```
The processed AIS file (`data/interim/ais_lalb_filtered.parquet`) and `data/raw/weather.csv` are included, so you do **not** need the huge raw files.

## Using your own raw data
* Put the daily NOAA/USCG files in `data/raw/` (`ais-2025-10-01.csv` ...) and delete `data/interim/ais_lalb_filtered.parquet`. `00_extract_raw_ais.py` rebuilds it and logs rows before/after.
* Delete `data/raw/weather.csv` to re-download it from Open-Meteo (`01_download_weather.py`, needs internet). The included file was recovered from your `hourly_table_full.csv` (same Open-Meteo values).
* Nothing is hard-coded: every threshold sits in `src/config.py`, every number in the dashboard and report is computed from the files.

## Project map
```
run_pipeline.py            runs steps 0-5
src/config.py              all settings (zones, thresholds, horizons, assumed demurrage rates)
src/00_extract_raw_ais.py  raw AIS -> cargo/tanker near LA/LB   (optional)
src/01_download_weather.py Open-Meteo wave + wind               (optional)
src/02_build_dataset.py    cleaning, geofences, hourly queue table, ship journeys, DBSCAN hotspots
src/03_train_models.py     rolling-origin validation, final models, SHAP, feature test
src/04_analytics.py        statistical tests, KPIs, 14 charts, decision + demurrage tables
src/05_make_report.py      report/PROJECT_REPORT.md and PRESENTATION_OUTLINE.md
src/features.py, common.py, charts.py, data_access.py, decisions.py   shared code
app/streamlit_app.py       dashboard (7 pages)
outputs/charts, tables, models     results
src/legacy/               your original notebook and scripts, kept for reference
```

## Dashboard pages
Overview (the 5-step story) · Port map (replay any hour, hotspots, zones) · Forecast (latest forecast, replay vs reality, what-if, model card with SHAP) · Alerts (late-arrival table, long-stay, queue surges) · Insights gallery (14 visuals, client + "so what") · Cost & decisions (demurrage calculator, action table) · Data & method.
The sidebar has a **replay clock**: pick any hour and the map, alerts and decisions update. This is the replacement for the optional live AISStream feed.

## What changed from the original idea (and why)
1. **A bigger target.** Only about 22 ships had a clean anchorage-to-berth wait. The pipeline now tracks every ship from port entry to berth (123 arrivals, 18 delayed), which supports real statistics.
2. **A forecast that is tested honestly.** The earlier models lost to "same as 3 hours ago". Validation is now rolling-origin, and the result is reported as found: the model only wins at 24 h ahead (about 30% lower error), so that is the horizon the dashboard promotes.
3. **Simple inputs.** Berth count, weekday and weather did not improve forecasts, so the final model uses the queue's history and the forecast hour. The tests are kept in `outputs/tables/feature_ablation.csv`.
4. **Local time.** Charts use Pacific time (AIS is UTC).
5. Dropped per your scope list: Optuna, Isolation Forest, live feed. PortWatch data was not supplied.

