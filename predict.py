import joblib
from datetime import datetime
import pandas as pd

# Load saved model
model = joblib.load(
    "models/indore_rainfall_model.joblib"
)

print("Model loaded successfully.")

# Current date and time
now = datetime.now()

# Weather inputs
temperature = float(input("Enter temperature (°C):"))
humidity = float(input("Enter relative humidity (%):"))
wind_speed = float(input("Enter wind speed (km/h): "))
cloud_cover = float(input("Enter cloud cover (%): "))
pressure = float(input("Enter pressure (hPa): "))

# Time features
hour = now.hour
day_of_week = now.weekday()
month = now.month

print("\nInput recieved successfully.")

print("Hour:", hour)
print("Day of week:", day_of_week)
print("Month:", month)

# Create input DataFrame
input_data = pd.DataFrame([{
    "temperature_2m": temperature,
    "relative_humidity_2m": humidity,
    "wind_speed_10m": wind_speed,
    "cloud_cover": cloud_cover,
    "pressure_msl": pressure,
    "hour": hour,
    "day_of_week": day_of_week,
    "month": month
}])

# Predict rain probability
rain_probability = model.predict_proba(input_data)[0][1]

# Convert to percentage
rain_probability_percent = rain_probability * 100

print("\n================================")
print("INDORE RAINFALL PREDICTION")
print("================================")

# Interpret probability
if rain_probability_percent < 30:
    chance = "Low"
elif rain_probability_percent <60:
    chance = "Moderate"
else:
    chance = "High"

print(f"Rain Probability: {rain_probability_percent:.2f}%")
print(f"Chance: {chance}")