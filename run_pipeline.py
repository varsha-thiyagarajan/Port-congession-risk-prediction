"""Run the whole project end to end:  python run_pipeline.py
Steps 0 and 1 are skipped automatically when their output already exists (no raw files / no internet needed)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
STEPS = [
    ("00_extract_raw_ais.py", "Extract cargo/tanker AIS near LA/LB from raw daily files (optional)"),
    ("01_download_weather.py", "Download wave + wind data (optional, needs internet)"),
    ("02_build_dataset.py", "Clean, geofence, hourly table, journeys, hotspots"),
    ("03_train_models.py", "Train + validate forecasting models, SHAP"),
    ("04_analytics.py", "Statistics, KPIs and the 14 story charts"),
    ("05_make_report.py", "Write the report and presentation outline"),
]
skip_if = {"00_extract_raw_ais.py": ROOT / "data/interim/ais_lalb_filtered.parquet",
           "01_download_weather.py": ROOT / "data/raw/weather.csv"}

for script, label in STEPS:
    if script in skip_if and skip_if[script].exists():
        print(f"[skip] {script}: {skip_if[script].name} already exists")
        continue
    print(f"\n[run ] {script}: {label}")
    r = subprocess.run([sys.executable, str(SRC / script)], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"Step {script} failed")
print("\nDone. Start the dashboard with:  streamlit run app/streamlit_app.py")
