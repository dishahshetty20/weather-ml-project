
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "live" / "live_weather_readings.csv"
OUTPUT_FILE = ROOT / "data" / "live" / "live_weather_30min_average.csv"

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    if df.empty or "collected_at" not in df.columns:
        raise ValueError("No collected weather readings are available.")

    df["collected_at"] = pd.to_datetime(df["collected_at"], errors="coerce")
    now = datetime.now(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None)
    cutoff = now - pd.Timedelta(minutes=30)

    recent = df.loc[
        (df["collected_at"] >= cutoff) &
        (df["collected_at"] <= now)
    ].copy()

    features = ["temperature", "humidity", "pressure", "wind_speed"]
    for col in features:
        recent[col] = pd.to_numeric(recent[col], errors="coerce")

    # Avoid giving repeated observations extra weight.
    recent = recent.sort_values("collected_at")
    recent = recent.drop_duplicates(
    subset=["observation_time"],
    keep="last"
    ) if "observation_time" in recent.columns else recent

    averages = recent[features].mean().to_dict()
    result = pd.DataFrame([{
        "calculated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "window_minutes": 30,
        "readings_count": len(recent),
        **{f"avg_{key}": value for key, value in averages.items()},
    }])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print("30-minute rolling average calculation completed.")
    print("Readings in window:", len(recent))
    print(result.to_string(index=False))
    print("Saved:", OUTPUT_FILE)

    if len(recent) < 2:
        print("WARNING: Not enough readings to validate a reliable rolling average.")

if __name__ == "__main__":
    main()
