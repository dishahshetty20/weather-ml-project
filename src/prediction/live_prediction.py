import requests
import pandas as pd
import joblib
import psycopg2

from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.save_live_prediction import save_prediction# =========================================================
# CONFIGURATION
# =========================================================

LATITUDE = 12.9716
LONGITUDE = 77.5946
TIMEZONE = "Asia/Kolkata"

API_URL = "https://api.open-meteo.com/v1/forecast"

MODEL_DIR = Path("models")
OUTPUT_DIR = Path("data/live")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# POSTGRESQL CONFIGURATION
# =========================================================

import os

DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = 5432
DB_NAME = "weather_db"
DB_USER = "weather_user"
DB_PASSWORD = "weather_password"


# =========================================================
# FEATURES USED DURING MODEL TRAINING
# =========================================================

FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "weather_code",
    "pressure_msl",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "wind_speed_10m",
    "wind_gusts_10m",
    "vapour_pressure_deficit"
]

# =========================================================
# FETCH WEATHER DATA
# =========================================================

def fetch_weather():

    print("Fetching live weather data from Open-Meteo...")

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(FEATURES),
        "forecast_days": 1,
        "timezone": TIMEZONE
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    print("API request successful!")

    return data["hourly"]


# =========================================================
# PREPARE LATEST WEATHER RECORD
# =========================================================

def prepare_latest_record(hourly_data):

    df = pd.DataFrame(hourly_data)

    df["time"] = pd.to_datetime(df["time"])

    df = df.sort_values("time").reset_index(drop=True)

    # Current time in Bengaluru timezone
    current_time = (
        pd.Timestamp.now(tz=TIMEZONE)
        .tz_localize(None)
    )

    # Find forecast hour closest to current time
    time_difference = abs(df["time"] - current_time)

    closest_index = time_difference.idxmin()

    latest = df.loc[[closest_index]].copy()

    return latest


# =========================================================
# LOAD TRAINED MODELS
# =========================================================

def load_models():

    print("\nLoading trained models...")

    rf_model = joblib.load(
        MODEL_DIR / "random_forest_rain_model.pkl"
    )

    xgb_model = joblib.load(
        MODEL_DIR / "xgboost_rain_model.pkl"
    )

    print("Models loaded successfully!")

    return rf_model, xgb_model


# =========================================================
# SAVE PREDICTION TO POSTGRESQL
# =========================================================

def save_prediction_to_database(
    prediction_time,
    rf_prediction,
    rf_probability,
    xgb_prediction,
    xgb_probability
):

    print("\nSaving prediction to PostgreSQL...")

    connection = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

    cursor = connection.cursor()

    insert_query = """
    INSERT INTO live_predictions (
        prediction_time,
        rf_prediction,
        rf_probability,
        xgb_prediction,
        xgb_probability
    )
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (prediction_time) DO NOTHING;
    """

    cursor.execute(
        insert_query,
        (
            prediction_time,
            int(rf_prediction),
            float(rf_probability),
            int(xgb_prediction),
            float(xgb_probability)
        )
    )

    connection.commit()

    if cursor.rowcount == 1:
        print("New prediction successfully saved to PostgreSQL!")
    else:
        print("Prediction already exists. Duplicate not inserted.")

    cursor.close()
    connection.close()

# =========================================================
# MAKE LIVE PREDICTION
# =========================================================

def make_prediction():

    # -----------------------------------------------------
    # Fetch weather
    # -----------------------------------------------------

    hourly_data = fetch_weather()

    # -----------------------------------------------------
    # Get current/closest weather record
    # -----------------------------------------------------

    latest = prepare_latest_record(hourly_data)

    print("\nLatest weather record:")

    print(
        latest[
            ["time"] + FEATURES
        ].to_string(index=False)
    )

    # -----------------------------------------------------
    # Prepare ML input
    # -----------------------------------------------------

    X_live = latest[FEATURES].astype("float64")

    # -----------------------------------------------------
    # Load models
    # -----------------------------------------------------

    rf_model, xgb_model = load_models()

    # -----------------------------------------------------
    # Random Forest prediction
    # -----------------------------------------------------

    rf_prediction = rf_model.predict(X_live)[0]

    rf_probability = (
        rf_model.predict_proba(X_live)[0][1]
    )

    # -----------------------------------------------------
    # XGBoost prediction
    # -----------------------------------------------------

    xgb_prediction = xgb_model.predict(X_live)[0]

    xgb_probability = (
        xgb_model.predict_proba(X_live)[0][1]
    )

    # -----------------------------------------------------
    # Prediction time
    # -----------------------------------------------------

    prediction_time = latest["time"].iloc[0]

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    print("\n" + "=" * 60)
    print("LIVE RAIN PREDICTION")
    print("=" * 60)

    print(
        f"Prediction time: {prediction_time}"
    )

    # -----------------------------------------------------
    # Random Forest result
    # -----------------------------------------------------

    print("\nRandom Forest:")

    print(
        "Prediction: "
        + (
            "RAIN"
            if rf_prediction == 1
            else "NO RAIN"
        )
    )

    print(
        f"Rain probability: "
        f"{rf_probability:.2%}"
    )

    # -----------------------------------------------------
    # XGBoost result
    # -----------------------------------------------------

    print("\nXGBoost:")

    print(
        "Prediction: "
        + (
            "RAIN"
            if xgb_prediction == 1
            else "NO RAIN"
        )
    )

    print(
        f"Rain probability: "
        f"{xgb_probability:.2%}"
    )

    # =====================================================
    # SAVE PREDICTION
    # =====================================================

    result = {
        "prediction_time": str(prediction_time),
        "rf_prediction": int(rf_prediction),
        "rf_probability": float(rf_probability),
        "xgb_prediction": int(xgb_prediction),
        "xgb_probability": float(xgb_probability),
        "created_at": datetime.now().isoformat()
    }

    result_df = pd.DataFrame([result])

    output_file = (
        OUTPUT_DIR / "live_predictions.csv"
    )

    # Append if file already exists
    if output_file.exists():

        result_df.to_csv(
            output_file,
            mode="a",
            header=False,
            index=False
        )

    # Create new file
    else:

        result_df.to_csv(
            output_file,
            index=False
        )

    print("\nPrediction saved to:")
    print(output_file)

    print("\nLIVE PREDICTION COMPLETED!")
    # =====================================================
    # SAVE TO POSTGRESQL
    # =====================================================

    save_prediction_to_database(
        prediction_time,
        rf_prediction,
        rf_probability,
        xgb_prediction,
        xgb_probability
    )

    print("\nLIVE PREDICTION COMPLETED!")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    make_prediction()