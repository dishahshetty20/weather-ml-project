from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Project root directory
ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data" / "raw" / "bengaluru_weather_2021_2025.csv"
OUTPUT_DIR = ROOT / "evaluation" / "data_profile"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    if not DATA_FILE.exists():
        print(f"Dataset not found: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE)

    # Convert timestamp columns where available
    time_col = next(
        (c for c in df.columns if c.lower() in ["time", "timestamp", "date"]),
        None
    )

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    print("\n========== WEATHER DATA PROFILE ==========")
    print(f"Dataset: {DATA_FILE.name}")
    print(f"Total records: {len(df)}")
    print(f"Total attributes: {len(df.columns)}")

    if time_col:
        print(f"Start date: {df[time_col].min()}")
        print(f"End date: {df[time_col].max()}")

    print("\n========== ATTRIBUTE INVENTORY ==========")
    inventory = pd.DataFrame({
        "Attribute": df.columns,
        "Data Type": [str(df[c].dtype) for c in df.columns],
        "Missing Values": [int(df[c].isna().sum()) for c in df.columns],
        "Unique Values": [int(df[c].nunique()) for c in df.columns]
    })

    print(inventory.to_string(index=False))
    inventory.to_csv(OUTPUT_DIR / "attribute_inventory.csv", index=False)

    # Descriptive statistics for numeric weather attributes
    numeric = df.select_dtypes(include="number")
    stats = numeric.describe().T
    stats["missing_values"] = numeric.isna().sum()
    stats.to_csv(OUTPUT_DIR / "descriptive_statistics.csv")

    print("\n========== DESCRIPTIVE STATISTICS ==========")
    print(stats.round(2).to_string())

    # Plot selected numerical weather attributes
    preferred = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "pressure_msl",
        "wind_speed_10m",
        "cloud_cover"
    ]

    selected = [c for c in preferred if c in df.columns]

    for column in selected:
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            continue

        plt.figure(figsize=(8, 4))
        plt.hist(values, bins=30)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{column}_distribution.png", dpi=150)
        plt.close()

    print(f"\nReports and graphs saved in:\n{OUTPUT_DIR}")
    print("Data profiling completed successfully.")

if __name__ == "__main__":
    main()