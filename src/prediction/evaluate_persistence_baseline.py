import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA = "data/processed/bengaluru_weather_processed.csv"
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
df = pd.read_csv(DATA)
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

# Match the regression training script's exact split.
split = int((len(df) - 1) * 0.80)
results = []

for target in TARGETS:
    actual = df[target].shift(-1)
    predicted = df[target]
    test_mask = (df.index >= split) & actual.notna()

    actual = actual[test_mask]
    predicted = predicted[test_mask]

    results.append({
        "target": target,
        "baseline": "Persistence",
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
        "test_rows": len(actual),
    })

result = pd.DataFrame(results)
print(result.to_string(index=False))
result.to_csv(
    "evaluation/regression/persistence_baseline.csv",
    index=False,
)
print("\nSaved aligned baseline.")
