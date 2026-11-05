
import requests

params = {
    "latitude": 33.7,
    "longitude": -118.3,
    "start_date": "2025-10-01",
    "end_date": "2025-10-14",
    "timezone": "UTC"
}

sources = {
    "WAVE HEIGHT": (
        "https://marine-api.open-meteo.com/v1/marine",
        "wave_height"
    ),
    "WIND SPEED": (
        "https://archive-api.open-meteo.com/v1/archive",
        "wind_speed_10m"
    )
}

for name, (url, variable) in sources.items():
    print("\n---", name, "---")
    p = {**params, "hourly": variable}

    try:
        r = requests.get(url, params=p, timeout=60)
        print("HTTP status:", r.status_code)
        r.raise_for_status()
        data = r.json()

        print("Timezone:", data.get("timezone"))
        print("UTC offset:", data.get("utc_offset_seconds"))
        print("Variable units:", data.get("hourly_units", {}))
        print("Returned rows:",
              len(data.get("hourly", {}).get("time", [])))

    except Exception as e:
        print("Verification failed:", e)
