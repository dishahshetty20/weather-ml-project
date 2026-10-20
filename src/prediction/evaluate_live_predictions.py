
import os
from pathlib import Path
from datetime import timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import psycopg2
import requests
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DB_HOST = os.getenv("DB_HOST", "postgres")
DB_NAME = os.getenv("DB_NAME", "weather_db")
DB_USER = os.getenv("DB_USER", "weather_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "weather_password")
DB_PORT = int(os.getenv("DB_PORT", "5432"))

LATITUDE = 12.9716
LONGITUDE = 77.5946
TIMEZONE = "Asia/Kolkata"

OUTPUT_DIR = ROOT / "evaluation" / "live_classification"
METRICS_FILE = OUTPUT_DIR / "live_classification_metrics.csv"
CONFUSION_MATRIX_FILE = OUTPUT_DIR / "live_confusion_matrices.png"
METRICS_GRAPH_FILE = OUTPUT_DIR / "live_classification_metrics.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT,
        connect_timeout=15,
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
        return cursor.fetchall()


# ============================================================
# FETCH ACTUAL RAIN FROM OPEN-METEO ARCHIVE
# ============================================================

def fetch_actual_weather(prediction_time):
    actual_time = prediction_time + timedelta(hours=1)
    date_string = actual_time.strftime("%Y-%m-%d")

    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": "rain",
            "start_date": date_string,
            "end_date": date_string,
            "timezone": TIMEZONE,
        },
        timeout=30,
    )
    response.raise_for_status()

    hourly = response.json().get("hourly", {})
    times = hourly.get("time", [])
    rain_values = hourly.get("rain", [])

    target_time = actual_time.strftime("%Y-%m-%dT%H:%M")

    for time_value, rain_value in zip(times, rain_values):
        if time_value == target_time:
            if rain_value is None:
                return None

            # Binary classification: 0 = no rain, 1 = rain.
            return 1 if float(rain_value) > 0 else 0

    return None


# ============================================================
# SAVE EVALUATION TO POSTGRESQL
# ============================================================

def save_evaluation(
    connection,
    prediction_time,
    actual_time,
    actual_rain,
    rf_prediction,
    xgb_prediction,
):
    rf_correct = int(rf_prediction == actual_rain)
    xgb_correct = int(xgb_prediction == actual_rain)

    query = """
        INSERT INTO live_prediction_evaluations (
            prediction_time,
            actual_time,
            actual_rain,
            rf_prediction,
            xgb_prediction,
            rf_correct,
            xgb_correct
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (prediction_time) DO NOTHING;
    """

    with connection:
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
                    xgb_correct,
                ),
            )


# ============================================================
# REPORT ACCUMULATED CLASSIFICATION METRICS
# ============================================================

def report_live_classification_metrics(connection):
    query = """
        SELECT
            actual_time,
            actual_rain,
            rf_prediction,
            xgb_prediction
        FROM live_prediction_evaluations
        ORDER BY actual_time;
    """

    with connection.cursor() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()

    if not rows:
        print("No evaluated live classification records yet.")
        return

    evaluation_df = pd.DataFrame(
        rows,
        columns=[
            "actual_time",
            "actual_rain",
            "rf_prediction",
            "xgb_prediction",
        ],
    )

    for column in [
        "actual_rain",
        "rf_prediction",
        "xgb_prediction",
    ]:
        evaluation_df[column] = pd.to_numeric(
            evaluation_df[column], errors="coerce"
        )

    evaluation_df = evaluation_df.dropna(
        subset=["actual_rain", "rf_prediction", "xgb_prediction"]
    )

    if evaluation_df.empty:
        print("No valid classification records available.")
        return

    actual = evaluation_df["actual_rain"].astype(int)
    rf_pred = evaluation_df["rf_prediction"].astype(int)
    xgb_pred = evaluation_df["xgb_prediction"].astype(int)

    results = []

    for model_name, predicted in [
        ("Random Forest", rf_pred),
        ("XGBoost", xgb_pred),
    ]:
        accuracy = accuracy_score(actual, predicted)
        precision = precision_score(
            actual, predicted, zero_division=0
        )
        recall = recall_score(
            actual, predicted, zero_division=0
        )
        f1 = f1_score(
            actual, predicted, zero_division=0
        )

        results.append({
            "model": model_name,
            "evaluated_samples": len(actual),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        })

        print(f"\n{model_name} — accumulated live metrics")
        print(f"Evaluated samples: {len(actual)}")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(METRICS_FILE, index=False)

    # --------------------------------------------------------
    # CONFUSION MATRIX GRAPH
    # --------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for axis, (model_name, predicted) in zip(
        axes,
        [
            ("Random Forest", rf_pred),
            ("XGBoost", xgb_pred),
        ],
    ):
        matrix = confusion_matrix(
            actual, predicted, labels=[0, 1]
        )

        axis.imshow(matrix, cmap="Blues")
        axis.set_title(f"{model_name} Confusion Matrix")
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("Actual class")
        axis.set_xticks([0, 1], ["No rain", "Rain"])
        axis.set_yticks([0, 1], ["No rain", "Rain"])

        for i in range(2):
            for j in range(2):
                text_color = (
                    "white" if matrix[i, j] > matrix.max() / 2
                    else "black"
                )
                axis.text(
                    j,
                    i,
                    str(matrix[i, j]),
                    ha="center",
                    va="center",
                    color=text_color,
                )

    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_FILE, dpi=150)
    plt.close(fig)

    # --------------------------------------------------------
    # METRIC COMPARISON GRAPH
    # --------------------------------------------------------

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    ]

    plot_df = metrics_df.set_index("model")[metric_columns]

    ax = plot_df.plot(
        kind="bar",
        figsize=(10, 6),
        ylim=(0, 1),
        title="Accumulated Live Rain Classification Metrics",
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", labelrotation=0)
    ax.legend(title="Metric")
    ax.grid(axis="y", alpha=0.3)

    fig = ax.figure
    fig.tight_layout()
    fig.savefig(METRICS_GRAPH_FILE, dpi=150)
    plt.close(fig)

    print("\nLive classification metrics saved:")
    print(f"CSV: {METRICS_FILE}")
    print(f"Confusion matrices: {CONFUSION_MATRIX_FILE}")
    print(f"Metric comparison graph: {METRICS_GRAPH_FILE}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("LIVE RAIN CLASSIFICATION EVALUATION")
    print("=" * 60)

    connection = get_connection()

    try:
        print("\nDatabase connection successful!")

        predictions = get_pending_predictions(connection)

        print(f"\nPending predictions: {len(predictions)}")

        from datetime import datetime
        now = datetime.now()

        evaluated_count = 0

        for row in predictions:
            prediction_time = row[0]
            rf_prediction = int(row[1])
            xgb_prediction = int(row[2])

            actual_time = prediction_time + timedelta(hours=1)

            print(f"\nPrediction time: {prediction_time}")
            print(f"Required actual time: {actual_time}")

            # Do not evaluate predictions whose target time is in the future.
            if actual_time > now:
                print("Actual weather is not available yet.")
                continue

            try:
                actual_rain = fetch_actual_weather(prediction_time)
            except Exception as error:
                print(f"Could not fetch actual weather: {error}")
                continue

            if actual_rain is None:
                print("Actual rain value unavailable.")
                continue

            print(f"Actual rain   : {actual_rain}")
            print(f"RF prediction : {rf_prediction}")
            print(f"XGB prediction: {xgb_prediction}")

            save_evaluation(
                connection,
                prediction_time,
                actual_time,
                actual_rain,
                rf_prediction,
                xgb_prediction,
            )

            print("Evaluation saved successfully!")
            evaluated_count += 1

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETED")
        print("=" * 60)
        print(f"New evaluations processed: {evaluated_count}")

        # Report accumulated metrics even when there are no new
        # predictions to evaluate during this run.
        report_live_classification_metrics(connection)

    finally:
        connection.close()
        print("\nDatabase connection closed.")


if __name__ == "__main__":
    main()
