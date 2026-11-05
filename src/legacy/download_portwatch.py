
import pandas as pd
from pathlib import Path

source = Path("Ports.csv")
output = Path("Data")
output.mkdir(exist_ok=True)

ports = pd.read_csv(source)

target = ports[
    ports["portname"].str.contains(
        "Los Angeles-Long Beach",
        case=False,
        na=False
    )
]

if target.empty:
    raise ValueError("Target port not found.")

target.to_csv(output / "portwatch.csv", index=False)

print("Matching ports:", len(target))
print("Saved to Data/portwatch.csv")
print(target[["portid", "portname", "fullname"]])
