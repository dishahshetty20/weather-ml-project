from pathlib import Path
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
    """Create the table and safely add the source column to existing databases."""
    create_query = """
        CREATE TABLE IF NOT EXISTS live_regression_evaluations (
            id BIGSERIAL PRIMARY KEY,
            prediction_for TIMESTAMP NOT NULL,
            target TEXT NOT NULL,
            method TEXT NOT NULL,
            predicted_value DOUBLE PRECISION NOT NULL,
            actual_value DOUBLE PRECISION NOT NULL,
            data_source TEXT NOT NULL DEFAULT 'legacy_unknown',
            error DOUBLE PRECISION NOT NULL,
            absolute_error DOUBLE PRECISION NOT NULL,
            evaluated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (prediction_for, target, method)
        );
    """

    migrate_query = """
        ALTER TABLE live_regression_evaluations
        ADD COLUMN IF NOT EXISTS data_source TEXT NOT NULL
        DEFAULT 'legacy_unknown';
    """

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(create_query)
            cur.execute(migrate_query)


def fetch_actuals(date_string):
    """Fetch hourly weather and label the source of the data."""
    common_params = {
        "latitude": 12.9716,
        "longitude": 77.5946,
        "hourly": ",".join(TARGETS),
        "timezone": "Asia/Kolkata",
    }

    sources = [
        (
            "https://archive-api.open-meteo.com/v1/archive",
            "archive",
            {
                **common_params,
                "start_date": date_string,
                "end_date": date_string,
            },
        ),
        (
            "https://api.open-meteo.com/v1/forecast",
            "forecast_model",
            {
                **common_params,
                "past_days": 1,
                "forecast_days": 1,
            },
        ),
    ]

    for url, source, params in sources:
        try:
            response = requests.get(url, params=params, timeout=30)

            if not response.ok:
                print(
                    f"{source} API returned HTTP {response.status_code} "
                    f"for {date_string}."
                )
                continue

            hourly = response.json().get("hourly", {})

            if not hourly or not hourly.get("time"):
                continue

            weather_frame = pd.DataFrame(hourly)
            weather_frame["time"] = pd.to_datetime(
                weather_frame["time"], errors="coerce"
            )
            weather_frame["data_source"] = source

            weather_frame = weather_frame[
                weather_frame["time"].dt.strftime("%Y-%m-%d")
                == date_string
            ].copy()

            if not weather_frame.empty:
                print(
                    f"Weather data for {date_string} retrieved from: "
                    f"{source}"
                )
                return weather_frame

        except requests.RequestException as exc:
            print(f"Could not fetch {date_string} from {source}: {exc}")
        except (ValueError, KeyError) as exc:
            print(f"Invalid weather response from {source}: {exc}")

    print(f"No weather data available for {date_string}.")
    return pd.DataFrame()


def load_predictions():
    """Load and standardize predictions from the live prediction CSV."""
    if not PREDICTIONS_FILE.exists():
        print(f"Predictions file not found: {PREDICTIONS_FILE}")
        return pd.DataFrame()

    predictions = pd.read_csv(PREDICTIONS_FILE)

    if predictions.empty:
        return predictions

    required = {
        "prediction_for",
        "target",
        "method",
        "predicted_value",
    }

    missing = required.difference(predictions.columns)

    if missing:
        raise ValueError(
            f"Prediction CSV is missing required columns: "
            f"{sorted(missing)}"
        )

    predictions["prediction_for"] = pd.to_datetime(
        predictions["prediction_for"], errors="coerce"
    )

    predictions["predicted_value"] = pd.to_numeric(
        predictions["predicted_value"], errors="coerce"
    )

    predictions = predictions.dropna(
        subset=[
            "prediction_for",
            "target",
            "method",
            "predicted_value",
        ]
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
    """Load existing evaluations without discarding saved results."""
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
        existing[column] = pd.to_numeric(
            existing[column], errors="coerce"
        )

    if "data_source" not in existing.columns:
        existing["data_source"] = "legacy_unknown"

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
    """Save evaluation rows to PostgreSQL."""
    if evaluations.empty:
        return

    rows = []

    for _, row in evaluations.iterrows():
        predicted = float(row["predicted_value"])
        actual = float(row["actual_value"])
        data_source = str(row.get("data_source", "legacy_unknown"))

        rows.append(
            (
                pd.Timestamp(row["prediction_for"]).to_pydatetime(),
                row["target"],
                row["method"],
                predicted,
                actual,
                data_source,
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
            data_source,
            error,
            absolute_error
        )
        VALUES %s
        ON CONFLICT (prediction_for, target, method)
        DO UPDATE SET
            predicted_value = EXCLUDED.predicted_value,
            actual_value = EXCLUDED.actual_value,
            data_source = EXCLUDED.data_source,
            error = EXCLUDED.error,
            absolute_error = EXCLUDED.absolute_error,
            evaluated_at = CURRENT_TIMESTAMP;
    """

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, query, rows)

    print(
        f"PostgreSQL: upserted {len(rows)} regression evaluation rows."
    )


def evaluate_pending_predictions(predictions, existing):
    """Evaluate past predictions when matching weather values are available."""
    if predictions.empty:
        print("No live regression predictions found.")
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

    dates = sorted(
        pending["prediction_for"].dt.strftime("%Y-%m-%d").unique()
    )

    actual_frames = {}

    for date_string in dates:
        actuals = fetch_actuals(date_string)

        if not actuals.empty:
            actual_frames[date_string] = actuals

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
                "data_source": matches.iloc[0].get(
                    "data_source", "archive"
                ),
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
    """Calculate MAE, RMSE and R2 by target, model, and data source."""
    if evaluations.empty:
        print("No evaluations available for metrics.")
        return pd.DataFrame()

    working = evaluations.copy()

    if "data_source" not in working.columns:
        working["data_source"] = "legacy_unknown"

    working["data_source"] = working["data_source"].fillna(
        "legacy_unknown"
    )

    records = []

    for (target, method, source), group in working.groupby(
        ["target", "method", "data_source"]
    ):
        actual = group["actual_value"].astype(float)
        predicted = group["predicted_value"].astype(float)

        record = {
            "target": target,
            "method": method,
            "data_source": source,
            "sample_count": len(group),
            "MAE": mean_absolute_error(actual, predicted),
            "RMSE": mean_squared_error(actual, predicted) ** 0.5,
            "R2": (
                r2_score(actual, predicted)
                if len(group) >= 2
                else float("nan")
            ),
        }

        records.append(record)

    metrics = pd.DataFrame(records)

    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_FILE, index=False)

    print(f"Live metrics saved to: {METRICS_FILE}")

    return metrics


def select_best_models(metrics):
    """Select the model with the lowest MAE for each target and source."""
    if metrics.empty:
        print("No metrics available to select best models.")
        return pd.DataFrame()

    best = (
        metrics.sort_values("MAE")
        .groupby(["target", "data_source"], as_index=False)
        .first()
    )

    BEST_MODELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    best.to_csv(BEST_MODELS_FILE, index=False)

    print(f"Best-model results saved to: {BEST_MODELS_FILE}")

    return best


def generate_live_plots(metrics):
    """Generate MAE comparison charts for live regression evaluation."""
    if metrics.empty:
        print("No metrics available for plotting.")
        return

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    for target in metrics["target"].unique():
        target_data = metrics[metrics["target"] == target]

        for source in target_data["data_source"].unique():
            plot_data = target_data[
                target_data["data_source"] == source
            ]

            if plot_data.empty:
                continue

            plt.figure(figsize=(8, 5))
            plt.bar(plot_data["method"], plot_data["MAE"])
            plt.title(f"Live MAE: {target} ({source})")
            plt.xlabel("Model")
            plt.ylabel(f"MAE ({UNITS.get(target, '')})")
            plt.xticks(rotation=15)
            plt.tight_layout()

            output_path = FIGURE_DIR / (
                f"live_mae_{target}_{source}.png"
            )

            plt.savefig(output_path, dpi=150)
            plt.close()

            print(f"Graph saved: {output_path}")

    # Actual-versus-predicted scatter plots for each weather variable.
    if EVALUATIONS_FILE.exists():
        try:
            evaluations = pd.read_csv(EVALUATIONS_FILE)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            evaluations = pd.DataFrame()

        if not evaluations.empty:
            for target in evaluations["target"].dropna().unique():
                target_data = evaluations[
                    evaluations["target"] == target
                ]

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
                            target_data["actual_value"],
                            errors="coerce",
                        ),
                        pd.to_numeric(
                            target_data["predicted_value"],
                            errors="coerce",
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
                ax.set_xlabel(
                    f"Actual value ({UNITS.get(target, '')})"
                )
                ax.set_ylabel(
                    f"Predicted value ({UNITS.get(target, '')})"
                )
                ax.legend()
                ax.grid(alpha=0.3)
                figure.tight_layout()

                safe_target = "".join(
                    character
                    if character.isalnum() or character in "_-"
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

    if "data_source" not in existing.columns:
        existing["data_source"] = "legacy_unknown"
    else:
        existing["data_source"] = existing["data_source"].fillna(
            "legacy_unknown"
        )

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

    metrics = calculate_metrics(combined)

    if not metrics.empty:
        print("\nLIVE REGRESSION METRICS")
        print(metrics.to_string(index=False))

        best_models = select_best_models(metrics)

        if not best_models.empty:
            print("\nBest live model by target (lowest MAE):")
            print(
                best_models[
                    ["target", "method", "MAE"]
                ].to_string(index=False)
            )
    else:
        print("No saved evaluations are available yet.")

    generate_live_plots(metrics)

    print(f"\nGraph folder: {FIGURE_DIR}")
    print("LIVE REGRESSION EVALUATION COMPLETE")


if __name__ == "__main__":
    main()