import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

import matplotlib.pyplot as plt
import seaborn as sns


# ==================================================
# FILE PATHS
# ==================================================

INPUT_FILE = "data/processed/bengaluru_weather_processed.csv"

MODEL_DIR = Path("models")
FIGURE_DIR = Path("evaluation/figures")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# LOAD DATA
# ==================================================

print("Loading processed dataset...")

df = pd.read_csv(INPUT_FILE)

df["time"] = pd.to_datetime(df["time"])

print(f"Total records: {len(df)}")


# ==================================================
# SORT BY TIME
# ==================================================

df = df.sort_values("time").reset_index(drop=True)


# ==================================================
# DEFINE FEATURES AND TARGET
# ==================================================

target = "rain_next_hour"

# Do not use time directly as a feature.
# Do not use rain itself because rain_next_hour is derived
# from the future rain observation and using it can cause
# target leakage.

features = [
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

X = df[features]
y = df[target]


print(f"\nNumber of features: {len(features)}")

print("\nFeatures used:")
for feature in features:
    print("-", feature)


# ==================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ==================================================

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTrain/Test Split")
print("----------------")
print(f"Training records: {len(X_train)}")
print(f"Testing records : {len(X_test)}")

print(
    f"\nTraining period: "
    f"{df['time'].iloc[0]} to {df['time'].iloc[split_index - 1]}"
)

print(
    f"Testing period : "
    f"{df['time'].iloc[split_index]} to {df['time'].iloc[-1]}"
)


# ==================================================
# RANDOM FOREST
# ==================================================

print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
rf_probabilities = rf_model.predict_proba(X_test)[:, 1]


# ==================================================
# XGBOOST
# ==================================================

print("Training XGBoost...")

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_predictions = xgb_model.predict(X_test)
xgb_probabilities = xgb_model.predict_proba(X_test)[:, 1]


# ==================================================
# EVALUATION FUNCTION
# ==================================================

def evaluate_model(name, y_true, predictions, probabilities):

    accuracy = accuracy_score(y_true, predictions)

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

    auc = roc_auc_score(
        y_true,
        probabilities
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            predictions,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, predictions))

    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc
    }


# ==================================================
# EVALUATE BOTH MODELS
# ==================================================

rf_results = evaluate_model(
    "Random Forest",
    y_test,
    rf_predictions,
    rf_probabilities
)

xgb_results = evaluate_model(
    "XGBoost",
    y_test,
    xgb_predictions,
    xgb_probabilities
)


# ==================================================
# SAVE RESULTS
# ==================================================

results = pd.DataFrame([
    rf_results,
    xgb_results
])

results.to_csv(
    "evaluation/model_comparison.csv",
    index=False
)

print("\nModel comparison saved to:")
print("evaluation/model_comparison.csv")


# ==================================================
# ROC CURVE
# ==================================================

rf_fpr, rf_tpr, _ = roc_curve(
    y_test,
    rf_probabilities
)

xgb_fpr, xgb_tpr, _ = roc_curve(
    y_test,
    xgb_probabilities
)

plt.figure(figsize=(8, 6))

plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"Random Forest (AUC = {rf_results['roc_auc']:.3f})"
)

plt.plot(
    xgb_fpr,
    xgb_tpr,
    label=f"XGBoost (AUC = {xgb_results['roc_auc']:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Rain Prediction")

plt.legend()
plt.grid()

plt.savefig(
    FIGURE_DIR / "roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nROC curve saved to:")
print("evaluation/figures/roc_curve.png")


# ==================================================
# CONFUSION MATRICES
# ==================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(
    confusion_matrix(y_test, rf_predictions),
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=axes[0]
)

axes[0].set_title("Random Forest")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")


sns.heatmap(
    confusion_matrix(y_test, xgb_predictions),
    annot=True,
    fmt="d",
    cmap="Greens",
    ax=axes[1]
)

axes[1].set_title("XGBoost")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "confusion_matrices.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Confusion matrices saved to:")
print("evaluation/figures/confusion_matrices.png")


# ==================================================
# FEATURE IMPORTANCE - RANDOM FOREST
# ==================================================

importance = pd.DataFrame({
    "feature": features,
    "importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\nTop 10 Random Forest Features:")
print(importance.head(10))

importance.to_csv(
    "evaluation/random_forest_feature_importance.csv",
    index=False
)


# ==================================================
# SAVE MODELS
# ==================================================

joblib.dump(
    rf_model,
    MODEL_DIR / "random_forest_rain_model.pkl"
)

joblib.dump(
    xgb_model,
    MODEL_DIR / "xgboost_rain_model.pkl"
)

print("\nModels saved successfully!")

print(
    "Random Forest: "
    "models/random_forest_rain_model.pkl"
)

print(
    "XGBoost: "
    "models/xgboost_rain_model.pkl"
)

print("\nML TRAINING COMPLETED!")