import pandas as pd

# Path to raw dataset
file_path = "data/raw/bengaluru_weather_2021_2025.csv"

# Load dataset
df = pd.read_csv(file_path)

print("=" * 60)
print("WEATHER DATASET CHECK")
print("=" * 60)

# 1. Dataset shape
print("\n1. DATASET SHAPE")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# 2. Date range
print("\n2. DATE RANGE")
df["time"] = pd.to_datetime(df["time"])
print(f"Start: {df['time'].min()}")
print(f"End  : {df['time'].max()}")

# 3. Column names
print("\n3. COLUMNS")
for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

# 4. Missing values
print("\n4. MISSING VALUES")
missing_values = df.isnull().sum()

print(missing_values[missing_values > 0])

if missing_values.sum() == 0:
    print("No missing values found.")

# 5. Duplicate rows
print("\n5. DUPLICATE ROWS")
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")

# 6. Data types
print("\n6. DATA TYPES")
print(df.dtypes)

# 7. Basic statistics
print("\n7. BASIC STATISTICS")
print(df.describe())

# 8. First 5 records
print("\n8. FIRST 5 RECORDS")
print(df.head())

# 9. Last 5 records
print("\n9. LAST 5 RECORDS")
print(df.tail())

print("\n" + "=" * 60)
print("DATASET CHECK COMPLETED")
print("=" * 60)