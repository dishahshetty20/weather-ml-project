import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

file = Path("evaluation/regression/all_models_comparison.csv")
out = Path("evaluation/regression")
df = pd.read_csv(file)

for target, group in df.groupby("target"):
    plt.figure(figsize=(8, 5))
    plt.bar(group["method"], group["MAE"])
    plt.title(f"MAE Comparison - {target}")
    plt.ylabel("Mean Absolute Error (lower is better)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    name = target.replace("/", "_")
    plt.savefig(out / f"{name}_mae_comparison.png", dpi=200)
    plt.close()

print("MAE graphs saved in evaluation/regression/")
