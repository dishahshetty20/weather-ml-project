import pandas as pd
from pathlib import Path

# --------------------------------------------------
# File paths
# --------------------------------------------------

INPUT_FILE = "data/raw/bengaluru_weather_2021_2025.csv"
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "bengaluru_weather_processed.csv"

# --------------------------------------------------
# Load data
# --------------------------------------------------

print("Loading historical weather data...")

df = pd.read_csv(INPUT_FILE)

# Convert time column
df["time"] = pd.to_datetime(df["time"])

print(f"Original rows: {len(df)}")

# --------------------------------------------------
# Create target
# --------------------------------------------------
# Target = whether rain occurs during the NEXT hour
#
# Current row:
#       10:00 weather
#
# Target:
#       rain at 11:00
#
# This prevents the model from using future information
# as an input feature.

df["rain_next_hour"] = df["rain"].shift(-1)

# Convert target into binary classification
df["rain_next_hour"] = (df["rain_next_hour"] > 0).astype(int)

# Remove the final row because there is no next-hour
# observation available for it
df = df.iloc[:-1].copy()

# --------------------------------------------------
# Remove unnecessary columns
# --------------------------------------------------

# 'time' is useful for ordering but will not be used
# directly as an ML feature.
#
# 'rain_next_hour' is our target.

print("\nTarget distribution:")
print(df["rain_next_hour"].value_counts())

# --------------------------------------------------
# Create processed-data directory
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Save processed dataset
# --------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

print("\nPreprocessing completed successfully!")
print(f"Processed rows: {len(df)}")
print(f"Processed columns: {len(df.columns)}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nFinal columns:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())