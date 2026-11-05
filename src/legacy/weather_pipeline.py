
import requests
import pandas as pd
from pathlib import Path

folder = Path("Data")
folder.mkdir(exist_ok=True)

lat, lon = 33.7, -118.3
start, end = "2025-10-01", "2025-10-14"

def fetch(url, variables):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(variables),
        "timezone": "UTC"
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    if "hourly" not in data:
        raise ValueError(data)

    df = pd.DataFrame(data["hourly"])
    missing = set(variables) - set(df.columns)
    if missing:
        raise ValueError(f"Missing API variables: {missing}")
    return df

weather_url = "https://marine-api.open-meteo.com/v1/marine"
wind_url = "https://archive-api.open-meteo.com/v1/archive"

waves = fetch(weather_url, ["wave_height"])
wind = fetch(wind_url, ["wind_speed_10m"])

df = waves.merge(wind, on="time", how="outer")
df = df.rename(columns={
    "wave_height": "wave_height",
    "wind_speed_10m": "wind_speed"
})
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

expected = pd.date_range(
    "2025-10-01 00:00",
    "2025-10-14 23:00",
    freq="h"
)

print("Rows:", len(df))
print("Duplicates:", df["time"].duplicated().sum())
print("Missing timestamps:", len(expected.difference(
    pd.DatetimeIndex(df["time"])
)))
print("Missing values:\n", df.isna().sum())

if len(df) != 336 or not pd.DatetimeIndex(df["time"]).equals(expected):
    raise ValueError("Hourly timeline validation failed.")

df.to_csv(folder / "weather.csv", index=False)
print("Saved:", folder / "weather.csv")
print(df.head())
