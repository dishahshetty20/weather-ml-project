import pandas as pd

models = pd.read_csv("evaluation/regression/regression_model_comparison.csv")
baseline = pd.read_csv("evaluation/regression/persistence_baseline.csv")

models = models.rename(columns={"model": "method"})
baseline = baseline.rename(columns={"baseline": "method"})

combined = pd.concat([models, baseline], ignore_index=True)
combined = combined[
    ["target", "method", "MAE", "RMSE", "R2", "test_rows"]
]
combined.to_csv(
    "evaluation/regression/all_models_comparison.csv",
    index=False
)

print(combined.to_string(index=False))
print("\nSaved clean comparison.")
