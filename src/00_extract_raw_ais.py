"""
STEP 0 (optional, run once on your own machine).
Reads the daily NOAA/USCG AIS files you downloaded into data/raw/ and keeps only
cargo + tanker ships near Los Angeles / Long Beach.

    data/raw/ais-2025-10-01.csv  (or .csv.zst)  ...  ais-2025-10-14.csv
 -> data/interim/ais_lalb_filtered.parquet
 -> outputs/tables/extraction_log.csv      (rows before / after, per day)

If the parquet already exists you can skip this step.
Download: https://hub.marinecadastre.gov/pages/vesselcount  (AIS Vessel Traces, daily files)
"""
import glob
import pandas as pd
import config as C
COLS = ["mmsi", "base_date_time", "latitude", "longitude", "sog", "cog",
        "heading", "vessel_type", "status", "length", "width", "draft"]
def main():
    files = sorted(glob.glob(str(C.RAW / "ais-2025-10-*.csv*")))
    if not files:
        print("No raw AIS files in data/raw/ - skipping. "
              f"Using the existing {C.AIS_FILTERED.name}." if C.AIS_FILTERED.exists()
              else "No raw AIS files and no filtered parquet: nothing to do.")
        return
    parts, log = [], []
    for f in files:
        before = after = 0
        for chunk in pd.read_csv(f, usecols=COLS, chunksize=500_000):
            before += len(chunk)
            keep = chunk[chunk.latitude.between(*C.EXTRACT_BOX["lat"])
                         & chunk.longitude.between(*C.EXTRACT_BOX["lon"])
                         & chunk.vessel_type.between(70, 89)]
            after += len(keep)
            parts.append(keep)
        log.append((f.split("/")[-1], before, after))
        print(f, "| all rows:", before, "| kept:", after)
    df = pd.concat(parts)
    df["base_date_time"] = pd.to_datetime(df["base_date_time"])
    df.to_parquet(C.AIS_FILTERED)
    pd.DataFrame(log, columns=["file", "rows_in_file", "rows_kept_la_lb_cargo_tanker"]) \
        .to_csv(C.TABLES / "extraction_log.csv", index=False)
    print("Saved", C.AIS_FILTERED)
if __name__ == "__main__":
    main()
