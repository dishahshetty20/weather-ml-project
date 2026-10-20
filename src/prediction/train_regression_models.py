from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

DATA_FILE = Path("data/processed/bengaluru_weather_processed.csv")
MODEL_DIR = Path("models/regression")
OUTPUT_DIR = Path("evaluation/regression")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use the same 12 weather inputs as the existing project.
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


def main():
    print("=" * 65)
    print("NEXT-HOUR WEATHER REGRESSION")
    print("=" * 65)

    df = pd.read_csv(DATA_FILE)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    required = FEATURES + TARGETS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Shift targets so each row predicts the following hour.
    for target in TARGETS:
        df[f"target_{target}"] = df[target].shift(-1)

    target_columns = [f"target_{target}" for target in TARGETS]
    df = df.dropna(subset=FEATURES + target_columns).reset_index(drop=True)

    X = df[FEATURES]
    split = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split], X.iloc[split:]

    print(f"Records: {len(df)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Training ends: {df['time'].iloc[split - 1]}")
    print(f"Testing starts: {df['time'].iloc[split]}")
    print("\nNote: the last target is the next-hour value.")

    results = []

    for target in TARGETS:
        y = df[f"target_{target}"]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        models = {
            "Linear Regression": LinearRegression(),
            "XGBoost Regressor": XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
            ),
        }

        print(f"\n{'-' * 55}\nTarget: next-hour {target}")

        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)

            print(
                f"{name}: MAE={mae:.4f}, "
                f"RMSE={rmse:.4f}, R2={r2:.4f}"
            )

            results.append({
                "target": target,
                "model": name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "test_rows": len(y_test),
            })

            model_path = MODEL_DIR / f"{target}_{'linear' if name == 'Linear Regression' else 'xgboost'}.joblib"
            joblib.dump(
                {
                    "model": model,
                    "features": FEATURES,
                    "target": target,
                    "forecast_horizon_hours": 1,
                },
                model_path,
            )

    result_df = pd.DataFrame(results)
    output_file = OUTPUT_DIR / "regression_model_comparison.csv"
    result_df.to_csv(output_file, index=False)

    print(f"\nResults saved to: {output_file}")
    print("\nComparison (lower MAE/RMSE is better; higher R2 is better):")
    print(result_df.to_string(index=False))
    print("\nRegression training completed.")


if __name__ == "__main__":
    main()

