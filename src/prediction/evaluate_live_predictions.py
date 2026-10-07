import os

import requests
import psycopg2


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DB_HOST = os.getenv("DB_HOST", "postgres")
DB_NAME = os.getenv("DB_NAME", "weather_db")
DB_USER = os.getenv("DB_USER", "weather_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "weather_password")
DB_PORT = os.getenv("DB_PORT", "5432")


# ============================================================
# BENGALURU LOCATION
# ============================================================

LATITUDE = 12.9716
LONGITUDE = 77.5946


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():

    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )


# ============================================================
# GET PENDING PREDICTIONS
# ============================================================

def get_pending_predictions(connection):

    query = """
       SELECT DISTINCT ON (lp.prediction_time)
    lp.prediction_time,
    lp.rf_prediction,
    lp.xgb_prediction
FROM live_predictions lp
LEFT JOIN live_prediction_evaluations lpe
    ON lp.prediction_time = lpe.prediction_time
WHERE lpe.id IS NULL
ORDER BY lp.prediction_time, lp.id;
    """

    with connection.cursor() as cursor:

        cursor.execute(query)

        rows = cursor.fetchall()

    return rows


# ============================================================
# FETCH ACTUAL RAIN
# ============================================================

def fetch_actual_weather(prediction_time):

    actual_time = prediction_time + \
        __import__("datetime").timedelta(hours=1)

    date_string = actual_time.strftime("%Y-%m-%d")

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "rain",
        "start_date": date_string,
        "end_date": date_string,
        "timezone": "Asia/Kolkata"
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    hourly = data.get("hourly", {})

    times = hourly.get("time", [])
    rain_values = hourly.get("rain", [])

    target_time = actual_time.strftime(
        "%Y-%m-%dT%H:%M"
    )

    for time_value, rain_value in zip(
        times,
        rain_values
    ):

        if time_value == target_time:

            if rain_value is None:
                return None

            return 1 if float(rain_value) > 0 else 0

    return None


# ============================================================
# SAVE EVALUATION
# ============================================================

def save_evaluation(
    connection,
    prediction_time,
    actual_time,
    actual_rain,
    rf_prediction,
    xgb_prediction
):

    rf_correct = int(
        rf_prediction == actual_rain
    )

    xgb_correct = int(
        xgb_prediction == actual_rain
    )

    query = """
        INSERT INTO live_prediction_evaluations
        (
            prediction_time,
            actual_time,
            actual_rain,
            rf_prediction,
            xgb_prediction,
            rf_correct,
            xgb_correct
        )
        VALUES
        (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s
        )
        ON CONFLICT (prediction_time) DO NOTHING;
    """

    with connection.cursor() as cursor:

        cursor.execute(
            query,
            (
                prediction_time,
                actual_time,
                actual_rain,
                rf_prediction,
                xgb_prediction,
                rf_correct,
                xgb_correct
            )
        )

    connection.commit()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("LIVE PREDICTION EVALUATION")
    print("=" * 60)

    connection = get_connection()

    print()
    print("Database connection successful!")

    try:

        predictions = get_pending_predictions(
            connection
        )

        print()
        print(
            f"Pending predictions: {len(predictions)}"
        )

        if not predictions:

            print()
            print(
                "No pending predictions."
            )

            return

        from datetime import datetime

        now = datetime.now()

        evaluated_count = 0

        for row in predictions:

            prediction_time = row[0]

            rf_prediction = int(row[1])

            xgb_prediction = int(row[2])

            actual_time = (
                prediction_time
                + __import__("datetime").timedelta(hours=1)
            )

            print()
            print(
                f"Prediction time: {prediction_time}"
            )

            print(
                f"Required actual time: {actual_time}"
            )

            # ------------------------------------------------
            # Do not evaluate future predictions
            # ------------------------------------------------

            if actual_time > now:

                print(
                    "Actual weather is not available yet."
                )

                continue

            # ------------------------------------------------
            # Get actual weather
            # ------------------------------------------------

            try:

                actual_rain = fetch_actual_weather(
                    prediction_time
                )

            except Exception as error:

                print(
                    f"Could not fetch actual weather: {error}"
                )

                continue

            if actual_rain is None:

                print(
                    "Actual rain value unavailable."
                )

                continue

            print(
                f"Actual rain   : {actual_rain}"
            )

            print(
                f"RF prediction : {rf_prediction}"
            )

            print(
                f"XGB prediction: {xgb_prediction}"
            )

            save_evaluation(
                connection,
                prediction_time,
                actual_time,
                actual_rain,
                rf_prediction,
                xgb_prediction
            )

            print(
                "Evaluation saved successfully!"
            )

            evaluated_count += 1

        print()
        print("=" * 60)
        print("EVALUATION COMPLETED")
        print("=" * 60)

        print(
            f"New evaluations processed: {evaluated_count}"
        )

    finally:

        connection.close()

        print()
        print(
            "Database connection closed."
        )


if __name__ == "__main__":
    main()