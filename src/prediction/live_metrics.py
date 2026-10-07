import os

import pandas as pd
import psycopg2

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DB_HOST = os.getenv("DB_HOST", "postgres")
DB_NAME = os.getenv("DB_NAME", "weather_db")
DB_USER = os.getenv("DB_USER", "weather_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "weather_password")
DB_PORT = os.getenv("DB_PORT", "5432")


# ============================================================
# CONNECT TO DATABASE
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
# LOAD EVALUATION DATA
# ============================================================

def load_evaluation_data(connection):

    query = """
        SELECT
            prediction_time,
            actual_rain,
            rf_prediction,
            xgb_prediction
        FROM live_prediction_evaluations
        ORDER BY prediction_time;
    """

    return pd.read_sql_query(
        query,
        connection
    )


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(name, y_true, predictions):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0
    )

    print()
    print(name)
    print("-" * len(name))

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")

    return {
        "model": name,
        "evaluated_predictions": len(y_true),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "macro_f1": macro_f1
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("LIVE MODEL EVALUATION METRICS")
    print("=" * 60)

    print()
    print("Connecting to PostgreSQL...")

    connection = get_connection()

    print("Database connection successful!")

    try:

        df = load_evaluation_data(
            connection
        )

        print()
        print(
            f"Evaluated predictions: {len(df)}"
        )

        if df.empty:

            print()
            print(
                "No evaluated predictions available yet."
            )

            return

        y_true = df["actual_rain"]

        rf_predictions = df[
            "rf_prediction"
        ]

        xgb_predictions = df[
            "xgb_prediction"
        ]

        rf_results = calculate_metrics(
            "Random Forest",
            y_true,
            rf_predictions
        )

        xgb_results = calculate_metrics(
            "XGBoost",
            y_true,
            xgb_predictions
        )

        # ----------------------------------------------------
        # Save live metrics
        # ----------------------------------------------------

        results = pd.DataFrame(
            [
                rf_results,
                xgb_results
            ]
        )

        output_file = (
            "evaluation/live_model_metrics.csv"
        )

        os.makedirs(
            "evaluation",
            exist_ok=True
        )

        results.to_csv(
            output_file,
            index=False
        )

        print()
        print(
            "Live metrics saved to:"
        )

        print(output_file)

        # ----------------------------------------------------
        # Display observation warning
        # ----------------------------------------------------

        if len(df) < 30:

            print()
            print(
                "NOTE: The current live evaluation "
                "sample is small."
            )

            print(
                "These metrics should be treated as "
                "preliminary live results."
            )

        print()
        print("=" * 60)
        print("LIVE METRICS CALCULATION COMPLETED")
        print("=" * 60)

    finally:

        connection.close()

        print()
        print(
            "Database connection closed."
        )


if __name__ == "__main__":
    main()
