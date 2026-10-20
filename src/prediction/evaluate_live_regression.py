from datetime import datetime
from pathlib import Path
import os

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_FILE = ROOT / "data" / "live" / "live_regression_predictions.csv"
EVALUATIONS_FILE = ROOT / "data" / "live" / "live_regression_evaluations.csv"
METRICS_FILE = ROOT / "evaluation" / "regression" / "live_metrics.csv"

LATITUDE = 12.9716
LONGITUDE = 77.5946
TIMEZONE = "Asia/Kolkata"

DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "weather_db")
DB_USER = os.getenv("DB_USER", "weather_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "weather_password")

TARGETS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "precipitation", "pressure_msl", "cloud_cover", "cloud_cover_low",
    "cloud_cover_mid", "wind_speed_10m", "wind_gusts_10m",
    "vapour_pressure_deficit",
]

UNITS = {
    "temperature_2m": "°C",
    "relative_humidity_2m": "%",
    "dew_point_2m": "°C",
    "precipitation": "mm",
    "pressure_msl": "hPa",
    "cloud_cover": "%",
    "cloud_cover_low": "%",
    "cloud_cover_mid": "%",
    "wind_speed_10m": "km/h",
    "wind_gusts_10m": "km/h",
    "vapour_pressure_deficit": "kPa",
}

COLUMNS = [
    "prediction_for", "target", "method", "predicted_value",
    "actual_value", "error", "absolute_error", "evaluated_at",
]

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS live_regression_evaluations (
    id BIGSERIAL PRIMARY KEY,
    prediction_for TIMESTAMP NOT NULL,
    target TEXT NOT NULL,
    method TEXT NOT NULL,
    predicted_value DOUBLE PRECISION NOT NULL,
    actual_value DOUBLE PRECISION NOT NULL,
    error DOUBLE PRECISION NOT NULL,
    absolute_error DOUBLE PRECISION NOT NULL,
    evaluated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_live_regression_evaluation
        UNIQUE (prediction_for, target, method)
);
"""

UPSERT_SQL = """
INSERT INTO live_regression_evaluations
    (prediction_for, target, method, predicted_value, actual_value,
     error, absolute_error, evaluated_at)
VALUES %s
ON CONFLICT (prediction_for, target, method)
DO UPDATE SET
    predicted_value = EXCLUDED.predicted_value,
    actual_value = EXCLUDED.actual_value,
    error = EXCLUDED.error,
    absolute_error = EXCLUDED.absolute_error,
    evaluated_at = EXCLUDED.evaluated_at;
"""


def normalize_time(value):
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def fetch_actuals(date_string):
    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": date_string,
            "end_date": date_string,
            "hourly": ",".join(TARGETS),
            "timezone": TIMEZONE,
        },
        timeout=30,
    )
    response.raise_for_status()
    hourly = response.json().get("hourly", {})
    if not hourly.get("time"):
        return {}
    actuals = pd.DataFrame(hourly)
    actuals["time_key"] = actuals["time"].map(normalize_time)
    return actuals.set_index("time_key").to_dict(orient="index")


def save_evaluations_to_postgres(frame):
    if frame.empty:
        print("No evaluation rows to write to PostgreSQL.")
        return

    rows = []
    for row in frame[COLUMNS].itertuples(index=False, name=None):
        prediction_for, target, method, predicted, actual, error, abs_error, evaluated_at = row
        rows.append((
            pd.to_datetime(prediction_for).to_pydatetime(),
            str(target),
            str(method),
            float(predicted),
            float(actual),
            float(error),
            float(abs_error),
            pd.to_datetime(evaluated_at).to_pydatetime(),
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
                print(f"PostgreSQL: upserted {len(rows)} regression evaluation rows.")
    finally:
        connection.close()


def main():
    print("=" * 60)
    print("LIVE REGRESSION EVALUATION")
    print("=" * 60)

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(f"Prediction file not found: {PREDICTIONS_FILE}")

    predictions = pd.read_csv(PREDICTIONS_FILE)
    if predictions.empty:
        print("No predictions available.")
        return

    required = {"prediction_for", "target", "method", "predicted_value", "created_at"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Prediction CSV is missing columns: {required - set(predictions.columns)}")

    predictions["time_key"] = predictions["prediction_for"].map(normalize_time)
    predictions["predicted_value"] = pd.to_numeric(predictions["predicted_value"], errors="coerce")
    predictions = predictions.dropna(subset=["time_key", "predicted_value"])
    predictions = predictions[predictions["target"].isin(TARGETS)]
    predictions = predictions.sort_values("created_at").drop_duplicates(
        subset=["time_key", "target", "method"], keep="last"
    )

    if EVALUATIONS_FILE.exists():
        evaluated = pd.read_csv(EVALUATIONS_FILE)
    else:
        evaluated = pd.DataFrame(columns=COLUMNS)

    if not evaluated.empty:
        for column in COLUMNS:
            if column not in evaluated.columns:
                evaluated[column] = pd.NA
        evaluated["time_key"] = evaluated["prediction_for"].map(normalize_time)
        evaluated = evaluated.dropna(subset=["time_key", "target", "method"])
        evaluated = evaluated.drop_duplicates(
            subset=["time_key", "target", "method"], keep="first"
        )
    else:
        evaluated["time_key"] = pd.Series(dtype=str)

    evaluated_keys = set()
    if not evaluated.empty:
        evaluated_keys = set(zip(evaluated["time_key"], evaluated["target"], evaluated["method"]))

    now = pd.Timestamp.now(tz=TIMEZONE).tz_localize(None)
    pending = predictions[predictions["time_key"].map(pd.to_datetime) < now].copy()
    pending = pending[
        ~pending.apply(
            lambda row: (row["time_key"], row["target"], row["method"]) in evaluated_keys,
            axis=1,
        )
    ]

    print(f"Unique predictions available: {len(predictions)}")
    print(f"Past predictions awaiting evaluation: {len(pending)}")

    actual_cache = {}
    new_rows = []
    unavailable = 0

    for _, row in pending.iterrows():
        target_time = pd.to_datetime(row["time_key"])
        date_string = target_time.strftime("%Y-%m-%d")
        target = row["target"]

        try:
            if date_string not in actual_cache:
                actual_cache[date_string] = fetch_actuals(date_string)
            actual_record = actual_cache[date_string].get(row["time_key"])
        except Exception as error:
            print(f"Archive request failed for {date_string}: {error}")
            unavailable += 1
            continue

        if actual_record is None or pd.isna(actual_record.get(target)):
            print(f"Actual unavailable: {target} at {row['time_key']}")
            unavailable += 1
            continue

        predicted = float(row["predicted_value"])
        actual = float(actual_record[target])
        new_rows.append({
            "prediction_for": row["time_key"],
            "target": target,
            "method": row["method"],
            "predicted_value": predicted,
            "actual_value": actual,
            "error": predicted - actual,
            "absolute_error": abs(predicted - actual),
            "evaluated_at": datetime.now().replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S"),
        })

    old_rows = evaluated.drop(columns=["time_key"], errors="ignore")
    if new_rows:
        evaluated = pd.concat([old_rows, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        evaluated = old_rows

    if evaluated.empty:
        print("No observations could be evaluated yet.")
        return

    evaluated["time_key"] = evaluated["prediction_for"].map(normalize_time)
    evaluated = evaluated.dropna(subset=["time_key", "target", "method"])
    evaluated = evaluated.drop_duplicates(
        subset=["time_key", "target", "method"], keep="first"
    )
    evaluated["prediction_for"] = evaluated["time_key"]
    evaluated = evaluated.drop(columns=["time_key"])[COLUMNS]

    EVALUATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    evaluated.to_csv(EVALUATIONS_FILE, index=False)

    # Sync the complete preserved evaluation CSV to PostgreSQL.
    # Upsert keys make repeated Airflow runs safe.
    save_evaluations_to_postgres(evaluated)

    metrics = []
    for (target, method), group in evaluated.groupby(["target", "method"]):
        actual = pd.to_numeric(group["actual_value"], errors="coerce")
        predicted = pd.to_numeric(group["predicted_value"], errors="coerce")
        valid = actual.notna() & predicted.notna()
        actual = actual[valid]
        predicted = predicted[valid]
        if len(actual) == 0:
            continue
        metrics.append({
            "target": target,
            "method": method,
            "MAE": mean_absolute_error(actual, predicted),
            "RMSE": mean_squared_error(actual, predicted) ** 0.5,
            "R2": r2_score(actual, predicted) if len(actual) >= 2 and actual.nunique() > 1 else float("nan"),
            "evaluated_rows": len(actual),
            "unit": UNITS[target],
        })

    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(METRICS_FILE, index=False)

    print(f"\nNew evaluation rows this run: {len(new_rows)}")
    print(f"Predictions without available actuals: {unavailable}")
    print(f"Total evaluation records in CSV: {len(evaluated)}")
    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"Evaluations saved to: {EVALUATIONS_FILE}")
    print(f"Metrics saved to: {METRICS_FILE}")


if __name__ == "__main__":
    main()
