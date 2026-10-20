from pathlib import Path
from datetime import datetime, timedelta
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_FILE = ROOT / "data" / "live" / "live_regression_predictions.csv"
EVALUATIONS_FILE = ROOT / "data" / "live" / "live_regression_evaluations.csv"
METRICS_FILE = ROOT / "evaluation" / "regression" / "live_metrics.csv"
FIGURE_DIR = ROOT / "evaluation" / "regression" / "figures"
BEST_MODELS_FILE = ROOT / "evaluation" / "regression" / "live_best_model_by_target.csv"

TARGETS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "pressure_msl",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "wind_speed_10m",
    "wind_gusts_10m",
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

METHODS = ["Linear Regression", "XGBoost Regressor"]


def get_db_connection():
    """Connect to the PostgreSQL database used by Airflow."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "weather_db"),
        user=os.getenv("POSTGRES_USER", "weather_user"),
        password=os.getenv("POSTGRES_PASSWORD", "weather_password"),
    )


def create_table():
    """Create the live regression evaluation table if it does not exist."""
    query = """
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
            UNIQUE (prediction_for, target, method)
        );
    """

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)


def fetch_actuals(date_string):
    """Fetch hourly observed weather values from Open-Meteo Archive."""
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": 12.9716,
        "longitude": 77.5946,
        "start_date": date_string,
        "end_date": date_string,
        "hourly": ",".join(TARGETS),
        "timezone": "Asia/Kolkata",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    payload = response.json()
    hourly = payload.get("hourly", {})

    if not hourly or not hourly.get("time"):
        return pd.DataFrame()

    actuals = pd.DataFrame(hourly)
    actuals["time"] = pd.to_datetime(actuals["time"])
    return actuals


def load_predictions():
    """Load and standardize predictions from the live prediction CSV."""
    if not PREDICTIONS_FILE.exists():
        print(f"Predictions file not found: {PREDICTIONS_FILE}")
        return pd.DataFrame()

    predictions = pd.read_csv(PREDICTIONS_FILE)

    if predictions.empty:
        return predictions

    required = {"prediction_for", "target", "method", "predicted_value"}
    missing = required.difference(predictions.columns)

    if missing:
        raise ValueError(
            f"Prediction CSV is missing required columns: {sorted(missing)}"
        )

    predictions["prediction_for"] = pd.to_datetime(
        predictions["prediction_for"], errors="coerce"
    )
    predictions["predicted_value"] = pd.to_numeric(
        predictions["predicted_value"], errors="coerce"
    )

    predictions = predictions.dropna(
        subset=["prediction_for", "target", "method", "predicted_value"]
    )

    predictions = predictions[
        predictions["target"].isin(TARGETS)
        & predictions["method"].isin(METHODS)
    ]

    return predictions.drop_duplicates(
        subset=["prediction_for", "target", "method"],
        keep="last",
    )


def load_existing_evaluations():
    """Load previous evaluation rows so historical metrics are retained."""
    if not EVALUATIONS_FILE.exists():
        return pd.DataFrame()

    try:
        existing = pd.read_csv(EVALUATIONS_FILE)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()

    if existing.empty:
        return existing

    required = {
        "prediction_for",
        "target",
        "method",
        "predicted_value",
        "actual_value",
    }

    if not required.issubset(existing.columns):
        return pd.DataFrame()

    existing["prediction_for"] = pd.to_datetime(
        existing["prediction_for"], errors="coerce"
    )

    for column in ["predicted_value", "actual_value"]:
        existing[column] = pd.to_numeric(existing[column], errors="coerce")

    existing = existing.dropna(
        subset=[
            "prediction_for",
            "target",
            "method",
            "predicted_value",
            "actual_value",
        ]
    )

    return existing


def save_evaluations_to_database(evaluations):
    """Insert evaluation rows without duplicating existing observations."""
    if evaluations.empty:
        return

    rows = []

    for _, row in evaluations.iterrows():
        predicted = float(row["predicted_value"])
        actual = float(row["actual_value"])

        rows.append(
            (
                pd.Timestamp(row["prediction_for"]).to_pydatetime(),
                row["target"],
                row["method"],
                predicted,
                actual,
                predicted - actual,
                abs(predicted - actual),
            )
        )

    query = """
        INSERT INTO live_regression_evaluations (
            prediction_for,
            target,
            method,
            predicted_value,
            actual_value,
            error,
            absolute_error
        )
        VALUES %s
        ON CONFLICT (prediction_for, target, method)
        DO UPDATE SET
            predicted_value = EXCLUDED.predicted_value,
            actual_value = EXCLUDED.actual_value,
            error = EXCLUDED.error,
            absolute_error = EXCLUDED.absolute_error,
            evaluated_at = CURRENT_TIMESTAMP;
    """

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, query, rows)

    print(f"PostgreSQL: upserted {len(rows)} regression evaluation rows.")


def evaluate_pending_predictions(predictions, existing):
    """Evaluate past predictions that do not already have actual values."""
    if predictions.empty:
        return pd.DataFrame(), 0

    existing_keys = set()

    if not existing.empty:
        existing_keys = set(
            zip(
                existing["prediction_for"],
                existing["target"],
                existing["method"],
            )
        )

    now = pd.Timestamp.now()
    pending = predictions[
        predictions["prediction_for"] < now
    ].copy()

    pending = pending[
        ~pending.apply(
            lambda row: (
                row["prediction_for"],
                row["target"],
                row["method"],
            ) in existing_keys,
            axis=1,
        )
    ]

    print(f"Unique predictions available: {len(predictions)}")
    print(f"Past predictions awaiting evaluation: {len(pending)}")

    if pending.empty:
        return pd.DataFrame(), 0

    pending["actual_value"] = float("nan")
    dates = sorted(
        pending["prediction_for"].dt.strftime("%Y-%m-%d").unique()
    )

    actual_frames = {}

    for date_string in dates:
        try:
            actuals = fetch_actuals(date_string)
            if not actuals.empty:
                actual_frames[date_string] = actuals
        except requests.RequestException as exc:
            print(f"Could not fetch actuals for {date_string}: {exc}")
        except (ValueError, KeyError) as exc:
            print(f"Invalid actual weather data for {date_string}: {exc}")

    evaluated_rows = []

    for _, row in pending.iterrows():
        date_string = row["prediction_for"].strftime("%Y-%m-%d")
        actuals = actual_frames.get(date_string)

        if actuals is None or actuals.empty:
            continue

        target = row["target"]
        if target not in actuals.columns:
            continue

        matches = actuals[
            actuals["time"] == row["prediction_for"]
        ]

        if matches.empty:
            continue

        actual_value = matches.iloc[0][target]

        if pd.isna(actual_value):
            continue

        evaluated_rows.append(
            {
                "prediction_for": row["prediction_for"],
                "target": target,
                "method": row["method"],
                "predicted_value": float(row["predicted_value"]),
                "actual_value": float(actual_value),
            }
        )

    new_evaluations = pd.DataFrame(evaluated_rows)
    unavailable_count = len(pending) - len(new_evaluations)

    if unavailable_count:
        print(
            f"Predictions without available actuals: {unavailable_count}"
        )

    return new_evaluations, len(pending)


def calculate_metrics(evaluations):
    """Calculate MAE, RMSE and R² for each target and model."""
    results = []

    if evaluations.empty:
        return pd.DataFrame()

    for (target, method), group in evaluations.groupby(["target", "method"]):
        actual = group["actual_value"].astype(float)
        predicted = group["predicted_value"].astype(float)

        mae = mean_absolute_error(actual, predicted)
        rmse = mean_squared_error(actual, predicted) ** 0.5

        # R² is not meaningful for a single observation.
        r2 = r2_score(actual, predicted) if len(group) >= 2 else float("nan")

        results.append(
            {
                "target": target,
                "method": method,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "evaluated_rows": len(group),
                "unit": UNITS.get(target, ""),
            }
        )

    return pd.DataFrame(results)


def generate_graphs(metrics_df):
    """Create MAE, RMSE and R² comparison graphs."""
    if metrics_df.empty:
        print("No metrics available; graphs were not generated.")
        return

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    for metric in ["MAE", "RMSE", "R2"]:
        pivot = metrics_df.pivot(
            index="target",
            columns="method",
            values=metric,
        )

        ax = pivot.plot(
            kind="bar",
            figsize=(15, 7),
            title=f"Live Regression Model Comparison - {metric}",
        )

        ax.set_xlabel("Weather variable")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", labelrotation=45)
        ax.legend(title="Model")
        ax.grid(axis="y", alpha=0.3)

        figure = ax.figure
        figure.tight_layout()
        output_file = FIGURE_DIR / f"live_{metric.lower()}_comparison.png"
        figure.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(figure)

        print(f"Graph saved: {output_file}")

    # Actual vs predicted scatter plots for each target.
    # These show how closely predictions match observed values.
    if EVALUATIONS_FILE.exists():
        try:
            evaluations = pd.read_csv(EVALUATIONS_FILE)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            evaluations = pd.DataFrame()

        if not evaluations.empty:
            for target in evaluations["target"].dropna().unique():
                target_data = evaluations[evaluations["target"] == target]

                if target_data.empty:
                    continue

                figure, ax = plt.subplots(figsize=(7, 6))

                for method, group in target_data.groupby("method"):
                    ax.scatter(
                        group["actual_value"],
                        group["predicted_value"],
                        alpha=0.7,
                        label=method,
                    )

                values = pd.concat(
                    [
                        pd.to_numeric(
                            target_data["actual_value"], errors="coerce"
                        ),
                        pd.to_numeric(
                            target_data["predicted_value"], errors="coerce"
                        ),
                    ]
                ).dropna()

                if not values.empty:
                    minimum = values.min()
                    maximum = values.max()
                    ax.plot(
                        [minimum, maximum],
                        [minimum, maximum],
                        linestyle="--",
                        label="Perfect prediction",
                    )

                ax.set_title(f"Actual vs Predicted - {target}")
                ax.set_xlabel(f"Actual value ({UNITS.get(target, '')})")
                ax.set_ylabel(f"Predicted value ({UNITS.get(target, '')})")
                ax.legend()
                ax.grid(alpha=0.3)
                figure.tight_layout()

                safe_target = "".join(
                    character if character.isalnum() or character in "_-"
                    else "_"
                    for character in str(target)
                )

                figure.savefig(
                    FIGURE_DIR / f"actual_vs_predicted_{safe_target}.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(figure)


def main():
    print("\nLIVE REGRESSION EVALUATION")

    for directory in [
        PREDICTIONS_FILE.parent,
        EVALUATIONS_FILE.parent,
        METRICS_FILE.parent,
        FIGURE_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    create_table()

    predictions = load_predictions()
    existing = load_existing_evaluations()

    new_evaluations, _ = evaluate_pending_predictions(
        predictions,
        existing,
    )

    if not new_evaluations.empty:
        save_evaluations_to_database(new_evaluations)

    # Merge previous and newly calculated evaluation rows.
    combined = pd.concat(
        [existing, new_evaluations],
        ignore_index=True,
    )

    if not combined.empty:
        combined["prediction_for"] = pd.to_datetime(
            combined["prediction_for"], errors="coerce"
        )

        combined = combined.dropna(
            subset=[
                "prediction_for",
                "target",
                "method",
                "predicted_value",
                "actual_value",
            ]
        )

        combined = combined.drop_duplicates(
            subset=["prediction_for", "target", "method"],
            keep="last",
        )

        combined["error"] = (
            combined["predicted_value"] - combined["actual_value"]
        )
        combined["absolute_error"] = combined["error"].abs()

    combined.to_csv(EVALUATIONS_FILE, index=False)

    print(f"New evaluation rows this run: {len(new_evaluations)}")
    print(f"Total evaluation records in CSV: {len(combined)}")
    print(f"Evaluations saved to: {EVALUATIONS_FILE}")

    metrics_df = calculate_metrics(combined)
    metrics_df.to_csv(METRICS_FILE, index=False)

    if not metrics_df.empty:
        print("\nLIVE REGRESSION METRICS")
        print(metrics_df.to_string(index=False))

        # Choose the model with the lowest MAE for each target.
        best_models = (
            metrics_df.sort_values("MAE")
            .groupby("target", as_index=False)
            .first()
        )

        best_models.to_csv(BEST_MODELS_FILE, index=False)

        print("\nBest live model by weather variable (lowest MAE):")
        print(
            best_models[["target", "method", "MAE", "evaluated_rows"]]
            .to_string(index=False)
        )
        print(f"Best-model summary saved to: {BEST_MODELS_FILE}")

    print(f"Metrics saved to: {METRICS_FILE}")

    generate_graphs(metrics_df)

    print(f"\nGraph folder: {FIGURE_DIR}")
    print("LIVE REGRESSION EVALUATION COMPLETE")


if __name__ == "__main__":
    main()