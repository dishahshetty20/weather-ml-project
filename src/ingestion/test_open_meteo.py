import requests

# Bengaluru location
LATITUDE = 12.9716
LONGITUDE = 77.5946

# Open-Meteo API
url = "https://api.open-meteo.com/v1/forecast"

# Weather parameters
params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "current": (
        "temperature_2m,"
        "relative_humidity_2m,"
        "dew_point_2m,"
        "apparent_temperature,"
        "precipitation,"
        "rain,"
        "weather_code,"
        "pressure_msl,"
        "surface_pressure,"
        "cloud_cover,"
        "wind_speed_10m,"
        "wind_direction_10m,"
        "wind_gusts_10m"
    ),
    "timezone": "Asia/Kolkata"
}

# Send request to API
response = requests.get(url, params=params, timeout=30)

# Check for errors
response.raise_for_status()

# Convert response to JSON
data = response.json()

# Display result
print("API request successful!")
print("\nCurrent weather:")
print(data["current"])