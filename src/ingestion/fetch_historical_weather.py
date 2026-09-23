import requests
import pandas as pd
from pathlib import Path

# Bengaluru coordinates
LATITUDE = 12.9716
LONGITUDE = 77.5946

# Historical period
START_DATE = "2021-01-01"
END_DATE = "2025-12-31"

# Open-Meteo Historical Weather API
URL = "https://archive-api.open-meteo.com/v1/archive"

# Weather attributes
HOURLY_VARIABLES = (
    "temperature_2m,"
    "relative_humidity_2m,"
    "dew_point_2m,"
    "apparent_temperature,"
    "precipitation,"
    "rain,"
    "snowfall,"
    "weather_code,"
    "pressure_msl,"
    "surface_pressure,"
    "cloud_cover,"
    "cloud_cover_low,"
    "cloud_cover_mid,"
    "cloud_cover_high,"
    "wind_speed_10m,"
    "wind_direction_10m,"
    "wind_gusts_10m,"
    "et0_fao_evapotranspiration,"
    "vapour_pressure_deficit,"
    "sunshine_duration,"
    "soil_temperature_0_to_7cm,"
    "soil_temperature_7_to_28cm,"
    "soil_moisture_0_to_7cm,"
    "soil_moisture_7_to_28cm"
)

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": HOURLY_VARIABLES,
    "timezone": "Asia/Kolkata"
}

print("Requesting historical weather data...")
print(f"Location: Bengaluru")
print(f"Period: {START_DATE} to {END_DATE}")

response = requests.get(URL, params=params, timeout=120)
response.raise_for_status()

data = response.json()

print("API request successful!")

# Convert hourly JSON data into a DataFrame
df = pd.DataFrame(data["hourly"])

# Create raw data directory
output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

# Save as CSV
output_file = output_dir / "bengaluru_weather_2021_2025.csv"

df.to_csv(output_file, index=False)

print("\nHistorical data downloaded successfully!")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Saved to: {output_file}")

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())