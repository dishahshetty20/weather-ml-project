
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data" / "raw" / "bengaluru_weather_2021_2025.csv"
OUTPUT_DIR = ROOT / "evaluation" / "weather_classifier"
MODEL_DIR = ROOT / "models" / "weather_classifier"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Features available at the current hour.
# The target is the weather category in the next hour.
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
    "vapour_pressure_deficit",
]

CATEGORIES = ["Clear", "Cloudy", "Drizzle", "Rain"]


def map_weather_category(code):
    """Map Open-Meteo weather codes to general weather categories."""
    if pd.isna(code):
        return None

    code = int(code)

    if code == 0:
        return "Clear"

    if code in [1, 2, 3]:
        return "Cloudy"

    if code in [51, 53, 55]:
        return "Drizzle"

    if code in [61, 63, 65]:
        return "Rain"

    return None


def main():
    print("=" * 70)
    print("WEATHER CLASSIFICATION: MODEL TRAINING AND EVALUATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load and validate the dataset
    # ---------------------------------------------------------
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = FEATURES + ["weather_code", "time"]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required dataset columns: {missing_columns}"
        )

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # ---------------------------------------------------------
    # 2. Create the next-hour weather target
    # ---------------------------------------------------------
    df["current_category"] = df["weather_code"].apply(
        map_weather_category
    )

    # Shift by one row to forecast the following hour.
    df["target_category"] = df["current_category"].shift(-1)

    # Only keep consecutive hourly observations.
    next_time = df["time"].shift(-1)
    consecutive_hour = (
        (next_time - df["time"]) == pd.Timedelta(hours=1)
    )

    df.loc[~consecutive_hour, "target_category"] = None

    df = df.dropna(
        subset=FEATURES + ["target_category"]
    ).copy()

    df = df[df["target_category"].isin(CATEGORIES)].copy()

    if len(df) < 20:
        raise ValueError(
            "Not enough valid rows to train and evaluate the classifier."
        )

    print(f"Usable rows: {len(df):,}")
    print(f"Features: {len(FEATURES)}")
    print("\nTarget category distribution:")
    print(df["target_category"].value_counts())

    # Save the documented category mapping.
    mapping_df = pd.DataFrame({
        "category": CATEGORIES,
        "description": [
            "Clear sky",
            "Cloudy or partly cloudy",
            "Light drizzle",
            "Rain",
        ],
    })
    mapping_df.to_csv(
        OUTPUT_DIR / "category_mapping.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 3. Encode labels
    # ---------------------------------------------------------
    label_encoder = LabelEncoder()
    label_encoder.fit(CATEGORIES)

    X = df[FEATURES].astype(float)
    y = label_encoder.transform(df["target_category"])

    category_labels = list(range(len(label_encoder.classes_)))
    rain_label = int(label_encoder.transform(["Rain"])[0])

    # ---------------------------------------------------------
    # 4. Chronological train/test split
    # ---------------------------------------------------------
    split_index = int(len(df) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Training or test dataset is empty.")

    print(f"\nTraining rows: {len(X_train):,}")
    print(f"Testing rows:  {len(X_test):,}")
    print("\nTraining class distribution:")
    print(
        pd.Series(
            label_encoder.inverse_transform(y_train)
        ).value_counts()
    )

    # ---------------------------------------------------------
    # 5. Define models
    # ---------------------------------------------------------
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=len(CATEGORIES),
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        ),

        "XGBoost Balanced": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=len(CATEGORIES),
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []

    # ---------------------------------------------------------
    # 6. Train, evaluate, and save each model
    # ---------------------------------------------------------
    for name, model in models.items():
        print("\n" + "=" * 70)
        print(f"Training {name}...")
        print("=" * 70)

        # Apply sample weights only to the balanced XGBoost model.
        if name == "XGBoost Balanced":
            sample_weights = compute_sample_weight(
                class_weight="balanced",
                y=y_train,
            )

            model.fit(
                X_train,
                y_train,
                sample_weight=sample_weights,
            )
        else:
            model.fit(X_train, y_train)

        predictions = model.predict(X_test).astype(int)
        probabilities = model.predict_proba(X_test)

        # Macro one-vs-rest ROC-AUC.
        try:
            auc_macro_ovr = roc_auc_score(
                y_test,
                probabilities,
                labels=category_labels,
                multi_class="ovr",
                average="macro",
            )
        except ValueError as exc:
            print(f"ROC-AUC unavailable: {exc}")
            auc_macro_ovr = float("nan")

        # Rain-specific performance.
        rain_precision = precision_score(
            y_test,
            predictions,
            labels=[rain_label],
            average="macro",
            zero_division=0,
        )

        rain_recall = recall_score(
            y_test,
            predictions,
            labels=[rain_label],
            average="macro",
            zero_division=0,
        )

        rain_f1 = f1_score(
            y_test,
            predictions,
            labels=[rain_label],
            average="macro",
            zero_division=0,
        )

        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision_macro": precision_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "recall_macro": recall_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "f1_macro": f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "f1_weighted": f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            ),
            "roc_auc_macro_ovr": auc_macro_ovr,
            "rain_precision": rain_precision,
            "rain_recall": rain_recall,
            "rain_f1": rain_f1,
        }

        results.append(metrics)

        print("\nEvaluation metrics:")
        for metric_name, value in metrics.items():
            if metric_name != "model":
                print(f"{metric_name}: {value:.4f}")

        print("\nClassification report:")
        print(
            classification_report(
                y_test,
                predictions,
                labels=category_labels,
                target_names=label_encoder.classes_,
                zero_division=0,
            )
        )

        # Save the trained model and its metadata.
        safe_name = (
            name.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        model_path = MODEL_DIR / f"{safe_name}.joblib"

        joblib.dump(
            {
                "model": model,
                "features": FEATURES,
                "label_encoder": label_encoder,
                "categories": list(label_encoder.classes_),
            },
            model_path,
        )

        print(f"Saved model: {model_path}")

        # Save confusion matrix.
        fig, ax = plt.subplots(figsize=(7, 6))

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            labels=category_labels,
            display_labels=label_encoder.classes_,
            cmap="Blues",
            xticks_rotation=45,
            ax=ax,
        )

        ax.set_title(f"Confusion Matrix - {name}")
        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR / f"confusion_matrix_{safe_name}.png",
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

    # ---------------------------------------------------------
    # 7. Save model comparison results
    # ---------------------------------------------------------
    results_df = pd.DataFrame(results)

    results_path = OUTPUT_DIR / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(f"\nComparison saved to: {results_path}")

    # ---------------------------------------------------------
    # 8. Plot model comparison
    # ---------------------------------------------------------
    plot_metrics = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "roc_auc_macro_ovr",
        "rain_precision",
        "rain_recall",
        "rain_f1",
    ]

    for metric in plot_metrics:
        fig, ax = plt.subplots(figsize=(9, 5))

        results_df.plot(
            x="model",
            y=metric,
            kind="bar",
            legend=False,
            ax=ax,
        )

        ax.set_title(f"Weather Classifier Comparison: {metric}")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_xlabel("Model")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=15)

        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR / f"comparison_{metric}.png",
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

    # Save label encoder for reuse.
    joblib.dump(
        label_encoder,
        MODEL_DIR / "label_encoder.joblib",
    )

    print("\nAll models, metrics, and graphs have been saved.")
    print(f"Models folder: {MODEL_DIR}")
    print(f"Evaluation folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

