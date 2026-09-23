import requests
import pandas as pd
import joblib

from pathlib import Path
from datetime import datetime


# =========================================================
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
# FEATURES USED DURING MODEL TRAINING
# =========================================================

FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "weather_code",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "et0_fao_evapotranspiration",
    "vapour_pressure_deficit",
    "sunshine_duration",
    "soil_temperature_0_to_7cm",
    "soil_temperature_7_to_28cm",
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm"
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


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    make_prediction()