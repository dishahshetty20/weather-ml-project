from pathlib import Path
from datetime import datetime
import os
import shutil

import joblib
import pandas as pd
import requests
import psycopg2
from psycopg2.extras import execute_values

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "regression"
OUTPUT_DIR = ROOT / "data" / "live"
OUTPUT_FILE = OUTPUT_DIR / "live_regression_predictions.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LATITUDE = 12.9716
LONGITUDE = 77.5946
TIMEZONE = "Asia/Kolkata"

DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "weather_db")
DB_USER = os.getenv("DB_USER", "weather_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "weather_password")

FEATURES = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "precipitation", "weather_code", "pressure_msl", "cloud_cover",
    "cloud_cover_low", "cloud_cover_mid", "wind_speed_10m",
    "wind_gusts_10m", "vapour_pressure_deficit",
]

TARGETS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "precipitation", "pressure_msl", "cloud_cover", "cloud_cover_low",
    "cloud_cover_mid", "wind_speed_10m", "wind_gusts_10m",
    "vapour_pressure_deficit",
]

COLUMNS = [
    "observation_time", "prediction_for", "target", "method",
    "predicted_value", "created_at",
]

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS live_regression_predictions (
    id BIGSERIAL PRIMARY KEY,
    observation_time TIMESTAMP NOT NULL,
    prediction_for TIMESTAMP NOT NULL,
    target TEXT NOT NULL,
    method TEXT NOT NULL,
    predicted_value DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_live_regression_prediction
        UNIQUE (prediction_for, target, method)
);
"""

UPSERT_SQL = """
INSERT INTO live_regression_predictions
    (observation_time, prediction_for, target, method, predicted_value, created_at)
VALUES %s
ON CONFLICT (prediction_for, target, method)
DO UPDATE SET
    observation_time = EXCLUDED.observation_time,
    predicted_value = EXCLUDED.predicted_value,
    created_at = EXCLUDED.created_at;
"""


def save_predictions_to_postgres(frame):
    """Persist this run's predictions to PostgreSQL; raise errors so Airflow detects failures."""
    if frame.empty:
        print("No prediction rows to write to PostgreSQL.")
        return

    rows = []
    for row in frame[COLUMNS].itertuples(index=False, name=None):
        observation_time, prediction_for, target, method, predicted_value, created_at = row
        rows.append((
            pd.to_datetime(observation_time).to_pydatetime(),
            pd.to_datetime(prediction_for).to_pydatetime(),
            str(target),
            str(method),
            float(predicted_value),
            pd.to_datetime(created_at).to_pydatetime(),
        ))

    connection = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASSWORD, connect_timeout=15,
    )
    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(CREATE_TABLE_SQL)
                execute_values(cursor, UPSERT_SQL, rows)
                print(f"PostgreSQL: upserted {len(rows)} live regression prediction rows.")
    finally:
        connection.close()


def main():
    print("Fetching live weather data from Open-Meteo...")
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": ",".join(FEATURES),
            "forecast_days": 1,
            "timezone": TIMEZONE,
        },
        timeout=30,
    )
    response.raise_for_status()

    data = response.json().get("hourly", {})
    if not data or not data.get("time"):
        raise RuntimeError("Open-Meteo returned no hourly weather data.")

    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    now = pd.Timestamp.now(tz=TIMEZONE).tz_localize(None)
    df = df.dropna(subset=["time"])
    df = df[df["time"] <= now].sort_values("time")

    if df.empty:
        raise RuntimeError("No current or past hourly observation available.")

    latest = df.tail(1)
    observation_time = latest["time"].iloc[0]
    prediction_for = observation_time + pd.Timedelta(hours=1)
    X = latest[FEATURES].astype("float64")

    print(f"Observation time: {observation_time}")
    print(f"Prediction valid for: {prediction_for}")

    results = []
    created_at = datetime.now().replace(microsecond=0)

    for target in TARGETS:
        for method, filename in [
            ("Linear Regression", f"{target}_linear.joblib"),
            ("XGBoost Regressor", f"{target}_xgboost.joblib"),
        ]:
            model_path = MODEL_DIR / filename
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")

            bundle = joblib.load(model_path)
            model = bundle["model"]
            model_features = bundle["features"]
            missing = set(model_features) - set(X.columns)
            if missing:
                raise ValueError(f"Missing features for {filename}: {sorted(missing)}")

            prediction = float(model.predict(X[model_features])[0])
            results.append({
                "observation_time": observation_time.strftime("%Y-%m-%d %H:%M:%S"),
                "prediction_for": prediction_for.strftime("%Y-%m-%d %H:%M:%S"),
                "target": target,
                "method": method,
                "predicted_value": prediction,
                "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
            })
            print(f"{target} | {method}: {prediction:.2f}")

    new_df = pd.DataFrame(results, columns=COLUMNS)

    # Keep a one-time backup of the current CSV before modifying it.
    if OUTPUT_FILE.exists():
        backup_file = OUTPUT_DIR / "live_regression_predictions_backup.csv"
        if not backup_file.exists():
            shutil.copy2(OUTPUT_FILE, backup_file)
            print(f"Original CSV backed up to: {backup_file}")

        old_df = pd.read_csv(OUTPUT_FILE)
        if all(column in old_df.columns for column in COLUMNS):
            for column in ["observation_time", "prediction_for"]:
                raw = old_df[column].astype(str).str.strip()
                valid = raw.str.fullmatch(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}")
                old_df = old_df.loc[valid].copy()
                old_df[column] = pd.to_datetime(old_df[column], errors="coerce").dt.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            old_df["predicted_value"] = pd.to_numeric(old_df["predicted_value"], errors="coerce")
            old_df = old_df.dropna(
                subset=["observation_time", "prediction_for", "predicted_value"]
            )[COLUMNS]
        else:
            print("Old CSV schema unexpected; original preserved in backup.")
            old_df = pd.DataFrame(columns=COLUMNS)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined["created_at"] = combined["created_at"].astype(str)
    combined = combined.sort_values("created_at").drop_duplicates(
        subset=["prediction_for", "target", "method"], keep="last"
    )
    combined = combined[COLUMNS].sort_values(["prediction_for", "target", "method"])
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions generated this run: {len(new_df)}")
    print(f"Unique prediction records saved to CSV: {len(combined)}")
    print(f"Saved to: {OUTPUT_FILE}")

    # Persist this run's rows to PostgreSQL as well as CSV.
    save_predictions_to_postgres(new_df)
    print("Live regression prediction completed.")


if __name__ == "__main__":
    main()
