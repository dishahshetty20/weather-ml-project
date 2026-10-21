
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "raw" / "bengaluru_weather_2021_2025.csv"
MODEL_DIR = ROOT / "models" / "weather_classifier"
OUTPUT_DIR = ROOT / "evaluation" / "weather_classifier"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "wind_speed_10m",
    "pressure_msl",
    "relative_humidity_2m",
]

CATEGORIES = ["Smooth", "Harsh"]
HARSH_CODES = [51, 53, 55, 61, 63, 65]


def main():
    print("=" * 60)
    print("HARSH / SMOOTH WEATHER CLASSIFIER")
    print("=" * 60)

    df = pd.read_csv(DATA_FILE)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    required = FEATURES + ["weather_code"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Target is the weather in the next hour.
    next_time = df["time"].shift(-1)
    next_code = df["weather_code"].shift(-1)

    consecutive = (next_time - df["time"]) == pd.Timedelta(hours=1)
    df["target"] = "Smooth"
    df.loc[next_code.isin(HARSH_CODES), "target"] = "Harsh"
    df.loc[next_code.isna() | ~consecutive, "target"] = None

    df = df.dropna(subset=FEATURES + ["target"]).copy()
    df = df[df["weather_code"].notna()].copy()
    df = df[df["weather_code"].isin(
        [0, 1, 2, 3, 51, 53, 55, 61, 63, 65]
    )].copy()

    if len(df) < 20 or df["target"].nunique() < 2:
        raise ValueError("Not enough data or target classes to train.")

    X = df[FEATURES].astype(float)
    y = df["target"].map({"Smooth": 0, "Harsh": 1}).astype(int)

    split = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("Both classes must appear in train and test sets.")

    print(f"Usable rows: {len(df):,}")
    print(f"Features: {FEATURES}")
    print("\nTarget distribution:")
    print(df["target"].value_counts())
    print(f"\nTraining rows: {len(X_train):,}")
    print(f"Testing rows: {len(X_test):,}")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    weights = compute_sample_weight("balanced", y_train)
    model.fit(X_train, y_train, sample_weight=weights)

    predictions = model.predict(X_test).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print("\nTEST RESULTS")
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1-score:  {f1:.2%}")
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=CATEGORIES,
        zero_division=0,
    ))
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    results = pd.DataFrame([{
        "model": "XGBoost Balanced",
        "features": ", ".join(FEATURES),
        "accuracy": accuracy,
        "precision_harsh": precision,
        "recall_harsh": recall,
        "f1_harsh": f1,
        "test_rows": len(X_test),
    }])
    results.to_csv(OUTPUT_DIR / "binary_model_results.csv", index=False)

    # The live classifier already expects this filename and bundle format.
    model_path = MODEL_DIR / "xgboost_balanced.joblib"
    joblib.dump({
        "model": model,
        "features": FEATURES,
        "categories": CATEGORIES,
        "target_rule": "Next-hour rain/drizzle codes = Harsh; other supported codes = Smooth",
    }, model_path)

    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {OUTPUT_DIR / 'binary_model_results.csv'}")
    print("\nTraining and evaluation completed.")


if __name__ == "__main__":
    main()
