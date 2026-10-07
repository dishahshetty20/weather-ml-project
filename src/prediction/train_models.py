from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
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

from xgboost import XGBClassifier

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = "data/processed/bengaluru_weather_processed.csv"

MODEL_DIR = Path("models")
FIGURE_DIR = Path("evaluation/figures")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading processed dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Total records: {len(df)}")


# ============================================================
# 2. PREPARE DATA
# ============================================================

df["time"] = pd.to_datetime(df["time"])

df = df.sort_values("time").reset_index(drop=True)


# ============================================================
# 3. SELECT 12 FEATURES
# ============================================================

features = [
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


target = "rain_next_hour"


print()
print(f"Number of features: {len(features)}")

print()
print("Features used:")

for feature in features:
    print(f"- {feature}")


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = features + [target]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# 5. REMOVE MISSING VALUES
# ============================================================

df = df.dropna(
    subset=required_columns
).reset_index(drop=True)


X = df[features]

y = df[target].astype(int)


# ============================================================
# 6. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print()
print("Train/Test Split")
print("----------------")
print(f"Training records: {len(X_train)}")
print(f"Testing records : {len(X_test)}")

print()
print(
    f"Training period: "
    f"{df['time'].iloc[0]} to "
    f"{df['time'].iloc[split_index - 1]}"
)

print(
    f"Testing period : "
    f"{df['time'].iloc[split_index]} to "
    f"{df['time'].iloc[-1]}"
)


# ============================================================
# 7. TRAIN RANDOM FOREST
# ============================================================

print()
print("Training Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)


# ============================================================
# 8. TRAIN XGBOOST
# ============================================================

print("Training XGBoost...")

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train,
    y_train
)


# ============================================================
# 9. EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    print()
    print(name)
    print("-" * len(name))

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    print()
    print("Classification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "macro_f1": macro_f1,
        "roc_auc": auc
    }


# ============================================================
# 10. EVALUATE BOTH MODELS
# ============================================================

rf_results = evaluate_model(
    "Random Forest",
    rf_model,
    X_test,
    y_test
)

xgb_results = evaluate_model(
    "XGBoost",
    xgb_model,
    X_test,
    y_test
)


# ============================================================
# 11. MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame(
    [
        rf_results,
        xgb_results
    ]
)

comparison_file = Path(
    "evaluation/model_comparison.csv"
)

comparison_file.parent.mkdir(
    parents=True,
    exist_ok=True
)

comparison.to_csv(
    comparison_file,
    index=False
)

print()
print("Model comparison saved to:")
print(comparison_file)


# ============================================================
# 12. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame(
    {
        "feature": features,
        "importance": rf_model.feature_importances_
    }
)

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

feature_importance_file = Path(
    "evaluation/random_forest_feature_importance.csv"
)

feature_importance.to_csv(
    feature_importance_file,
    index=False
)

print()
print("Random Forest feature importance saved to:")
print(feature_importance_file)


# ============================================================
# 13. ROC CURVE
# ============================================================

rf_probabilities = rf_model.predict_proba(
    X_test
)[:, 1]

xgb_probabilities = xgb_model.predict_proba(
    X_test
)[:, 1]


rf_fpr, rf_tpr, _ = roc_curve(
    y_test,
    rf_probabilities
)

xgb_fpr, xgb_tpr, _ = roc_curve(
    y_test,
    xgb_probabilities
)


rf_auc = roc_auc_score(
    y_test,
    rf_probabilities
)

xgb_auc = roc_auc_score(
    y_test,
    xgb_probabilities
)


plt.figure(figsize=(8, 6))

plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"Random Forest (AUC = {rf_auc:.4f})"
)

plt.plot(
    xgb_fpr,
    xgb_tpr,
    label=f"XGBoost (AUC = {xgb_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve - Rain Prediction"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

roc_file = FIGURE_DIR / "roc_curve.png"

plt.savefig(
    roc_file,
    dpi=300
)

plt.close()

print()
print("ROC curve saved to:")
print(roc_file)


# ============================================================
# 14. CONFUSION MATRICES
# ============================================================

rf_predictions = rf_model.predict(
    X_test
)

xgb_predictions = xgb_model.predict(
    X_test
)


rf_cm = confusion_matrix(
    y_test,
    rf_predictions
)

xgb_cm = confusion_matrix(
    y_test,
    xgb_predictions
)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)


axes[0].imshow(
    rf_cm,
    interpolation="nearest"
)

axes[0].set_title(
    "Random Forest"
)

axes[0].set_xlabel(
    "Predicted"
)

axes[0].set_ylabel(
    "Actual"
)

axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])

for i in range(2):
    for j in range(2):
        axes[0].text(
            j,
            i,
            rf_cm[i, j],
            ha="center",
            va="center"
        )


axes[1].imshow(
    xgb_cm,
    interpolation="nearest"
)

axes[1].set_title(
    "XGBoost"
)

axes[1].set_xlabel(
    "Predicted"
)

axes[1].set_ylabel(
    "Actual"
)

axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])

for i in range(2):
    for j in range(2):
        axes[1].text(
            j,
            i,
            xgb_cm[i, j],
            ha="center",
            va="center"
        )


plt.suptitle(
    "Confusion Matrices - Rain Prediction"
)

plt.tight_layout()

confusion_file = FIGURE_DIR / "confusion_matrices.png"

plt.savefig(
    confusion_file,
    dpi=300
)

plt.close()

print()
print("Confusion matrices saved to:")
print(confusion_file)


# ============================================================
# 15. SAVE TRAINED MODELS
# ============================================================

rf_model_file = MODEL_DIR / "random_forest_rain_model.pkl"

xgb_model_file = MODEL_DIR / "xgboost_rain_model.pkl"


joblib.dump(
    rf_model,
    rf_model_file
)

joblib.dump(
    xgb_model,
    xgb_model_file
)


print()
print("Models saved:")
print(rf_model_file)
print(xgb_model_file)


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print()
print(f"Features used : {len(features)}")
print(f"Training rows : {len(X_train)}")
print(f"Testing rows  : {len(X_test)}")

print()
print("Evaluation metrics:")
print(
    "Accuracy, Precision, Recall, F1-score, "
    "Macro F1, ROC-AUC"
)

print()
print("Output files:")
print(f"- {comparison_file}")
print(f"- {feature_importance_file}")
print(f"- {roc_file}")
print(f"- {confusion_file}")
print(f"- {rf_model_file}")
print(f"- {xgb_model_file}")

print()
print("Done.")