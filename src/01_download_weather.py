"""
STEP 1 (optional). Downloads hourly wave height + wind speed from Open-Meteo for the
exact days found in the AIS file, and writes data/raw/weather.csv.
Needs internet. If data/raw/weather.csv already exists you can skip this step.
"""
import pandas as pd
import requests
import config as C
def fetch(url, variable, start, end):
    r = requests.get(url, params=dict(latitude=C.WEATHER_LAT, longitude=C.WEATHER_LON,
                                      start_date=start, end_date=end,
                                      hourly=variable, timezone="UTC"), timeout=60)
    r.raise_for_status()
    data = r.json()
    if "hourly" not in data:
        raise ValueError(data)
    return pd.DataFrame(data["hourly"])
def main():
    ais = pd.read_parquet(C.AIS_FILTERED, columns=["base_date_time"])
    start = ais.base_date_time.min().strftime("%Y-%m-%d")
    end = ais.base_date_time.max().strftime("%Y-%m-%d")
    waves = fetch("https://marine-api.open-meteo.com/v1/marine", "wave_height", start, end)
    wind = fetch("https://archive-api.open-meteo.com/v1/archive", "wind_speed_10m", start, end)
    w = waves.merge(wind, on="time", how="outer").rename(columns={"wind_speed_10m": "wind_speed"})
    w["time"] = pd.to_datetime(w["time"])
    w = w.sort_values("time").reset_index(drop=True)
    expected = pd.date_range(f"{start} 00:00", f"{end} 23:00", freq="h")
    assert pd.DatetimeIndex(w["time"]).equals(expected), "hourly timeline validation failed"
    w.to_csv(C.WEATHER_CSV, index=False)
    print("Saved", C.WEATHER_CSV, w.shape, "| missing:", w.isna().sum().to_dict())


if __name__ == "__main__":
    main()
