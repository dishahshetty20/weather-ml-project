
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import os
import pandas as pd
import requests
import psycopg2

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "live"
OUTPUT_FILE = OUTPUT_DIR / "live_weather_readings.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LATITUDE = 12.9716
LONGITUDE = 77.5946
TIMEZONE = "Asia/Kolkata"

DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "weather_db")
DB_USER = os.getenv("DB_USER", "weather_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "weather_password")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS live_weather_readings (
    id BIGSERIAL PRIMARY KEY,
    observation_time TIMESTAMP NOT NULL,
    collected_at TIMESTAMP NOT NULL,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    pressure DOUBLE PRECISION,
    wind_speed DOUBLE PRECISION,
    precipitation DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (collected_at)
);
"""


def main():
    print("Fetching live weather snapshot from Open-Meteo...")

    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "current": (
                "temperature_2m,relative_humidity_2m,"
                "pressure_msl,wind_speed_10m,precipitation"
            ),
            "timezone": TIMEZONE,
        },
        timeout=30,
    )
    response.raise_for_status()

    current = response.json().get("current")
    if not current or not current.get("time"):
        raise RuntimeError("No current weather data returned.")

    collected_at = datetime.now(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None)
    reading = {
        "observation_time": current["time"],
        "collected_at": collected_at.strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": current.get("temperature_2m"),
        "humidity": current.get("relative_humidity_2m"),
        "pressure": current.get("pressure_msl"),
        "wind_speed": current.get("wind_speed_10m"),
        "precipitation": current.get("precipitation"),
    }

    frame = pd.DataFrame([reading])

    if OUTPUT_FILE.exists():
        old = pd.read_csv(OUTPUT_FILE)
        frame = pd.concat([old, frame], ignore_index=True)

    frame["observation_time"] = pd.to_datetime(
        frame["observation_time"], errors="coerce"
    )
    frame["collected_at"] = pd.to_datetime(
        frame["collected_at"], errors="coerce"
    )

    frame = frame.dropna(subset=["observation_time", "collected_at"])
    frame = frame.drop_duplicates(
        subset=["collected_at"], keep="last"
    ).sort_values("collected_at")

    frame.to_csv(OUTPUT_FILE, index=False)

    connection = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        connect_timeout=15,
    )

    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(CREATE_TABLE_SQL)

                # Migrate an existing table safely.
                cursor.execute("""
                    ALTER TABLE live_weather_readings
                    ADD COLUMN IF NOT EXISTS collected_at TIMESTAMP;
                """)

                # Preserve old records by assigning their existing
                # observation timestamp as the best available fallback.
                cursor.execute("""
                    UPDATE live_weather_readings
                    SET collected_at = observation_time
                    WHERE collected_at IS NULL;
                """)

                cursor.execute("""
                    ALTER TABLE live_weather_readings
                    ALTER COLUMN collected_at SET NOT NULL;
                """)

                cursor.execute("""
                    ALTER TABLE live_weather_readings
                    DROP CONSTRAINT IF EXISTS live_weather_readings_observation_time_key;
                """)

                cursor.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS
                    live_weather_readings_collected_at_key
                    ON live_weather_readings (collected_at);
                """)

                cursor.execute("""
                    INSERT INTO live_weather_readings (
                        observation_time, collected_at, temperature,
                        humidity, pressure, wind_speed, precipitation
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (collected_at) DO NOTHING;
                """, (
                    pd.to_datetime(reading["observation_time"]).to_pydatetime(),
                    collected_at,
                    reading["temperature"],
                    reading["humidity"],
                    reading["pressure"],
                    reading["wind_speed"],
                    reading["precipitation"],
                ))

        print("Reading saved to PostgreSQL.")
    finally:
        connection.close()

    print("Observation time:", reading["observation_time"])
    print("Collection time:", reading["collected_at"])
    print("Temperature:", reading["temperature"], "°C")
    print("Humidity:", reading["humidity"], "%")
    print("Pressure:", reading["pressure"], "hPa")
    print("Wind speed:", reading["wind_speed"], "km/h")
    print("Saved CSV:", OUTPUT_FILE)
    print("Live weather collection completed.")


if __name__ == "__main__":
    main()
