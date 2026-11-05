"""Central settings. Every threshold lives here so nothing is hidden inside code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
CHARTS = OUT / "charts"
TABLES = OUT / "tables"
MODELS = OUT / "models"
for p in (RAW, INTERIM, PROCESSED, CHARTS, TABLES, MODELS):
    p.mkdir(parents=True, exist_ok=True)

# ---- input files -----------------------------------------------------------
AIS_FILTERED = INTERIM / "ais_lalb_filtered.parquet"   # output of 00_extract_raw_ais.py
WEATHER_CSV = RAW / "weather.csv"                      # output of 01_download_weather.py
PORTWATCH_CSV = RAW / "portwatch.csv"                  # optional (IMF PortWatch)

# ---- study area (lon_min, lon_max, lat_min, lat_max) ------------------------
EXTRACT_BOX = dict(lat=(33.40, 33.85), lon=(-118.60, -117.90))
BERTH = (-118.305, -118.190, 33.728, 33.780)
BERTH_W = (-118.305, -118.245, 33.700, 33.735)
ANCH_IN = (-118.240, -118.020, 33.640, 33.740)
ANCH_OUT = (-118.050, -118.000, 33.590, 33.630)
STATIONARY_KN = 1.0            # slower than this = stationary

# ---- data-quality rules ------------------------------------------------------
MAX_SOG_KN = 35.0              # faster than this is not a cargo ship / tanker
MAX_JUMP_KN = 45.0             # implied speed between two pings above this = GPS jump

# ---- waiting-time / visit logic ---------------------------------------------
LOW_COVERAGE_ROWS = 50         # hours with fewer AIS rows than this are "low coverage"
LONG_STAY_H = 72               # at anchor 3+ days = lay-up, not a queue
EPISODE_GAP_H = 12             # a gap longer than this starts a new episode
DELAY_THRESHOLD_H = 6.0        # port entry -> berth longer than this = "delayed arrival"
TZ_LOCAL = "America/Los_Angeles"

# ---- weather ----------------------------------------------------------------
WEATHER_LAT, WEATHER_LON = 33.7, -118.3

# ---- machine learning --------------------------------------------------------
HORIZONS = [3, 6, 12, 24]      # hours ahead
FINAL_HORIZON = 24             # the one the dashboard promotes
MIN_TRAIN_ROWS = 96            # first rolling-origin fold starts after this many hours
FOLD_HOURS = 24                # each validation fold = 1 day
RANDOM_STATE = 42

# ---- DBSCAN hotspots ---------------------------------------------------------
DBSCAN_EPS_KM = 0.5
DBSCAN_MIN_PTS = 10

# ---- demurrage scenario: ASSUMED values, editable in the dashboard ------------
DEMURRAGE_RATES_USD_DAY = [15_000, 30_000, 45_000]
REDUCTION_SCENARIOS = [0.10, 0.20]
