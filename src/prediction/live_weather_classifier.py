from pathlib import Path
from datetime import datetime
import os

import joblib
import pandas as pd
import requests
import psycopg2

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "weather_classifier" / "xgboost_balanced.joblib"
OUTPUT_DIR = ROOT / "data" / "live"
OUTPUT_FILE = OUTPUT_DIR / "live_weather_predictions.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LATITUDE = 12.9716
LONGITUDE = 77.5946
TIMEZONE = "Asia/Kolkata"

DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "weather_db")
DB_USER = os.getenv("DB_USER", "weather_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "weather_password")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS live_weather_class_predictions (
    id BIGSERIAL PRIMARY KEY,
    observation_time TIMESTAMP NOT NULL,
    prediction_for TIMESTAMP NOT NULL,
    predicted_category TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (prediction_for)
);
"""

UPSERT_SQL = """
INSERT INTO live_weather_class_predictions
    (observation_time, prediction_for, predicted_category, confidence, created_at)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (prediction_for) DO UPDATE SET
    observation_time = EXCLUDED.observation_time,
    predicted_category = EXCLUDED.predicted_category,
    confidence = EXCLUDED.confidence,
    created_at = EXCLUDED.created_at;
"""


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    features = bundle["features"]
    categories = [str(category) for category in bundle["categories"]]

    print("Fetching current weather from Open-Meteo...")
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": ",".join(features),
            "forecast_days": 1,
            "timezone": TIMEZONE,
        },
        timeout=30,
    )
    response.raise_for_status()

    hourly = response.json().get("hourly", {})
    if not hourly or not hourly.get("time"):
        raise RuntimeError("Open-Meteo returned no hourly weather data.")

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    now = pd.Timestamp.now(tz=TIMEZONE).tz_localize(None)
    df = df.dropna(subset=["time"])
    df = df[df["time"] <= now].sort_values("time")

    if df.empty:
        raise RuntimeError("No current or past hourly weather data is available.")

    latest = df.tail(1).copy()
    missing = [feature for feature in features if feature not in latest.columns]
    if missing:
        raise ValueError(f"Missing required model features: {missing}")

    if latest[features].isna().any().any():
        raise ValueError("Current weather contains missing model feature values.")

    X = latest[features].astype(float)
    probabilities = model.predict_proba(X)[0]
    predicted_index = int(probabilities.argmax())
    predicted_category = categories[predicted_index]
    confidence = float(probabilities[predicted_index])

    observation_time = latest["time"].iloc[0]
    prediction_for = observation_time + pd.Timedelta(hours=1)
    created_at = datetime.now()

    record = {
        "observation_time": observation_time.strftime("%Y-%m-%d %H:%M:%S"),
        "prediction_for": prediction_for.strftime("%Y-%m-%d %H:%M:%S"),
        "predicted_category": predicted_category,
        "confidence": confidence,
        "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
    }

    output_df = pd.DataFrame([record])
    if OUTPUT_FILE.exists():
        previous = pd.read_csv(OUTPUT_FILE)
        output_df = pd.concat([previous, output_df], ignore_index=True)

    output_df = output_df.drop_duplicates(
        subset=["prediction_for"], keep="last"
    ).sort_values("prediction_for")
    output_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Observation time: {record['observation_time']}")
    print(f"Forecast for: {record['prediction_for']}")
    print(f"Predicted category: {predicted_category}")
    print(f"Model confidence: {confidence:.2%}")
    print(f"CSV saved: {OUTPUT_FILE}")

    connection = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        connect_timeout=15,
    )
    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(CREATE_TABLE_SQL)
                cursor.execute(
                    UPSERT_SQL,
                    (
                        observation_time.to_pydatetime(),
                        prediction_for.to_pydatetime(),
                        predicted_category,
                        confidence,
                        created_at,
                    ),
                )
        print("Prediction saved to PostgreSQL.")
    finally:
        connection.close()

    print("Live general weather classification completed.")


if __name__ == "__main__":
    main()
