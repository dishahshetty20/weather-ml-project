
from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "bengaluru_weather_ml.csv"
MODEL_DIR = ROOT / "models" / "weather_multiclass"
OUT_DIR = ROOT / "evaluation" / "classification"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. Load and prepare data
# --------------------------------------------------
df = pd.read_csv(DATA)
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

features = [
    c for c in df.columns
    if c not in ["time", "weather_code"]
]

# Predict the next hour's weather code.
df["target"] = df["weather_code"].shift(-1)
df = df.dropna(subset=["target"]).copy()

X = df[features].apply(pd.to_numeric, errors="coerce")
valid = X.notna().all(axis=1)
df = df.loc[valid].copy()
X = X.loc[valid]
y_original = df["target"].astype(int)

encoder = LabelEncoder()
y = encoder.fit_transform(y_original)

split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y[:split], y[split:]

if len(set(y_train)) < 2:
    raise ValueError(
        "Training data contains fewer than two weather classes."
    )

# --------------------------------------------------
# 2. Define models
# --------------------------------------------------
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    ),
}

results = []
probabilities = {}
predictions = {}
class_labels = list(range(len(encoder.classes_)))

# --------------------------------------------------
# 3. Train, predict, evaluate and save models
# --------------------------------------------------
for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)

    probabilities[name] = prob
    predictions[name] = pred

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision_macro": precision_score(
            y_test, pred, average="macro", zero_division=0
        ),
        "Recall_macro": recall_score(
            y_test, pred, average="macro", zero_division=0
        ),
        "F1_macro": f1_score(
            y_test, pred, average="macro", zero_division=0
        ),
        "Precision_weighted": precision_score(
            y_test, pred, average="weighted", zero_division=0
        ),
        "Recall_weighted": recall_score(
            y_test, pred, average="weighted", zero_division=0
        ),
        "F1_weighted": f1_score(
            y_test, pred, average="weighted", zero_division=0
        ),
    }

    try:
        metrics["ROC_AUC_ovr_weighted"] = roc_auc_score(
            y_test,
            prob,
            multi_class="ovr",
            average="weighted",
            labels=class_labels,
        )
    except ValueError:
        metrics["ROC_AUC_ovr_weighted"] = float("nan")

    results.append(metrics)

    model_filename = (
        "random_forest.joblib"
        if name == "Random Forest"
        else "xgboost.joblib"
    )
    joblib.dump(model, MODEL_DIR / model_filename)

    # Per-class classification report
    report = classification_report(
        y_test,
        pred,
        labels=class_labels,
        target_names=[str(c) for c in encoder.classes_],
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        OUT_DIR / f"{name.lower().replace(' ', '_')}_report.csv"
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test, pred, labels=class_labels
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=encoder.classes_,
    )
    disp.plot(xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title(f"{name}: Next-Hour Weather-Code Prediction")
    plt.tight_layout()
    plt.savefig(
        OUT_DIR
        / f"{name.lower().replace(' ', '_')}_confusion_matrix.png",
        dpi=150,
    )
    plt.close()

# --------------------------------------------------
# 4. Save model comparison and comparison graph
# --------------------------------------------------
comparison = pd.DataFrame(results)
comparison.to_csv(OUT_DIR / "model_comparison.csv", index=False)

comparison.set_index("Model")[
    ["Accuracy", "Precision_macro", "Recall_macro", "F1_macro"]
].plot(kind="bar", ylim=(0, 1), figsize=(10, 6))

plt.ylabel("Score")
plt.title("Weather Classification Model Comparison")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT_DIR / "model_comparison.png", dpi=150)
plt.close()

# --------------------------------------------------
# 5. Multiclass ROC curves
# --------------------------------------------------
# Use only classes represented in the test set.
present_classes = sorted(set(y_test))
y_binary_present = label_binarize(
    y_test, classes=present_classes
)

if len(present_classes) >= 2:
    plt.figure(figsize=(9, 7))

    for name, prob in probabilities.items():
        # Select probability columns corresponding to present classes.
        prob_present = prob[:, present_classes]

        try:
            fpr, tpr, _ = roc_curve(
                y_binary_present.ravel(),
                prob_present.ravel(),
            )
            auc = roc_auc_score(
                y_binary_present,
                prob_present,
                average="micro",
            )
            plt.plot(
                fpr, tpr,
                label=f"{name} (micro-AUC={auc:.3f})",
            )
        except ValueError:
            continue

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC Curves (Micro-Average)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_curves.png", dpi=150)
    plt.close()

# --------------------------------------------------
# 6. Multiclass Precision-Recall curves (NEW)
# --------------------------------------------------
# Flatten the one-vs-rest labels and probabilities to create
# a micro-averaged precision-recall curve.
if len(present_classes) >= 2:
    plt.figure(figsize=(9, 7))

    for name, prob in probabilities.items():
        prob_present = prob[:, present_classes]

        precision, recall, _ = precision_recall_curve(
            y_binary_present.ravel(),
            prob_present.ravel(),
        )

        ap = average_precision_score(
            y_binary_present,
            prob_present,
            average="micro",
        )

        plt.plot(
            recall,
            precision,
            label=f"{name} (AP={ap:.3f})",
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Multiclass Precision-Recall Curves (Micro-Average)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUT_DIR / "precision_recall_curves.png",
        dpi=150,
    )
    plt.close()

# --------------------------------------------------
# 7. Identify best model by macro F1 and save summary
# --------------------------------------------------
best = comparison.sort_values(
    "F1_macro", ascending=False
).iloc[0]

summary = {
    "prediction_task": (
        "Predict next-hour weather_code "
        "from current-hour features"
    ),
    "dataset_rows_used": int(len(df)),
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "feature_names": features,
    "weather_code_classes": [
        int(c) for c in encoder.classes_
    ],
    "best_model_by_macro_f1": str(best["Model"]),
    "metrics": comparison.where(
        pd.notna(comparison), None
    ).to_dict("records"),
}

with open(OUT_DIR / "training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nMODEL COMPARISON")
print(comparison.round(4).to_string(index=False))
print(f"\nBest model by macro F1: {best['Model']}")
print(f"\nSaved models: {MODEL_DIR}")
print(f"Saved metrics and graphs: {OUT_DIR}")
print(
    "Precision-recall graph: "
    f"{OUT_DIR / 'precision_recall_curves.png'}"
)
