
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "raw" / "bengaluru_weather_2021_2025.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "bengaluru_weather_ml.csv"

FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
]

TARGET = "weather_code"

def main():
    df = pd.read_csv(INPUT_FILE)

    required = ["time"] + FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    selected = df[required].copy()
    selected["time"] = pd.to_datetime(selected["time"], errors="coerce")
    selected = selected.dropna(subset=required)
    selected = selected.sort_values("time").drop_duplicates(subset=["time"])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(OUTPUT_FILE, index=False)

    print("ML dataset prepared successfully.")
    print("Rows:", len(selected))
    print("Input weather attributes:", len(FEATURES))
    print("Target:", TARGET)
    print("Target categories:", selected[TARGET].nunique())
    print("\nMost common weather codes:")
    print(selected[TARGET].value_counts().head(10).to_string())
    print("\nSaved:", OUTPUT_FILE)

if __name__ == "__main__":
    main()
