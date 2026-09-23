import pandas as pd
import psycopg2

# --------------------------------------------------
# PostgreSQL connection details
# --------------------------------------------------

DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "weather_db"
DB_USER = "weather_user"
DB_PASSWORD = "weather_password"

# --------------------------------------------------
# CSV file
# --------------------------------------------------

CSV_FILE = "data/raw/bengaluru_weather_2021_2025.csv"

# --------------------------------------------------
# Load CSV
# --------------------------------------------------

print("Loading historical weather data...")

df = pd.read_csv(CSV_FILE)

df["time"] = pd.to_datetime(df["time"])

print(f"Rows loaded from CSV: {len(df)}")
print(f"Columns: {len(df.columns)}")

# --------------------------------------------------
# Connect to PostgreSQL
# --------------------------------------------------

print("\nConnecting to PostgreSQL...")

connection = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)

cursor = connection.cursor()

print("PostgreSQL connection successful!")

# --------------------------------------------------
# Create table
# --------------------------------------------------

create_table_query = """
CREATE TABLE IF NOT EXISTS historical_weather (
    time TIMESTAMP PRIMARY KEY,
    temperature_2m DOUBLE PRECISION,
    relative_humidity_2m INTEGER,
    dew_point_2m DOUBLE PRECISION,
    apparent_temperature DOUBLE PRECISION,
    precipitation DOUBLE PRECISION,
    rain DOUBLE PRECISION,
    snowfall DOUBLE PRECISION,
    weather_code INTEGER,
    pressure_msl DOUBLE PRECISION,
    surface_pressure DOUBLE PRECISION,
    cloud_cover INTEGER,
    cloud_cover_low INTEGER,
    cloud_cover_mid INTEGER,
    cloud_cover_high INTEGER,
    wind_speed_10m DOUBLE PRECISION,
    wind_direction_10m INTEGER,
    wind_gusts_10m DOUBLE PRECISION,
    et0_fao_evapotranspiration DOUBLE PRECISION,
    vapour_pressure_deficit DOUBLE PRECISION,
    sunshine_duration DOUBLE PRECISION,
    soil_temperature_0_to_7cm DOUBLE PRECISION,
    soil_temperature_7_to_28cm DOUBLE PRECISION,
    soil_moisture_0_to_7cm DOUBLE PRECISION,
    soil_moisture_7_to_28cm DOUBLE PRECISION
);
"""

cursor.execute(create_table_query)

connection.commit()

print("Table 'historical_weather' is ready.")

# --------------------------------------------------
# Insert data
# --------------------------------------------------

insert_query = """
INSERT INTO historical_weather (
    time,
    temperature_2m,
    relative_humidity_2m,
    dew_point_2m,
    apparent_temperature,
    precipitation,
    rain,
    snowfall,
    weather_code,
    pressure_msl,
    surface_pressure,
    cloud_cover,
    cloud_cover_low,
    cloud_cover_mid,
    cloud_cover_high,
    wind_speed_10m,
    wind_direction_10m,
    wind_gusts_10m,
    et0_fao_evapotranspiration,
    vapour_pressure_deficit,
    sunshine_duration,
    soil_temperature_0_to_7cm,
    soil_temperature_7_to_28cm,
    soil_moisture_0_to_7cm,
    soil_moisture_7_to_28cm
)
VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s
)
ON CONFLICT (time) DO NOTHING;
"""

print("\nInserting data into PostgreSQL...")

for row in df.itertuples(index=False, name=None):
    cursor.execute(insert_query, row)

connection.commit()

print("Historical data inserted successfully!")

# --------------------------------------------------
# Verify number of records
# --------------------------------------------------

cursor.execute("SELECT COUNT(*) FROM historical_weather;")

count = cursor.fetchone()[0]

print("\nDatabase verification:")
print(f"Records in PostgreSQL: {count}")

# --------------------------------------------------
# Close connection
# --------------------------------------------------

cursor.close()
connection.close()

print("\nPostgreSQL connection closed.")
print("Data loading completed successfully!")