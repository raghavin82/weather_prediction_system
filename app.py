import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# Load trained model
model = joblib.load(
    "models/indore_rainfall_model.joblib"
)

# Page configuration
st.set_page_config(
    page_title="Indore Rainfall Prediction",
    page_icon="🌧️",
    layout="centered"
)

# Sidebar
st.sidebar.title("About the Project")

st.sidebar.write(
    "This system predicts the probability of rainfall "
    "in Indore during the next hour."
)

st.sidebar.write("**Model:** Random Forest")
st.sidebar.write("**Calibration:** Sigmoid")
st.sidebar.write("**Prediction:** Next-hour rainfall")

# Title
st.title("🌧️ Indore Rainfall Prediction System")

# Description
st.write(
    "Predict the probability of rain in Indore during the next hour."
)

st.divider()

# Weather input section
st.subheader("Weather Conditions")

st.write("Enter the current weather conditions below.")

# Weather inputs
temperature = st.number_input(
    "Temperature (°C)",
    min_value=-10.0,
    max_value=50.0,
    value=25.0,
    step=0.1
)

humidity = st.number_input(
    "Relative Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=60.0,
    step=1.0
)

wind_speed = st.number_input(
    "Wind Speed (km/h)",
    min_value=0.0,
    max_value=100.0,
    value=10.0,
    step=0.1
)

cloud_cover = st.number_input(
    "Cloud Cover (%)",
    min_value=0.0,
    max_value=100.0,
    value=50.0,
    step=1.0
)

pressure = st.number_input(
    "Pressure (hPa)",
    min_value=950.0,
    max_value=1100.0,
    value=1010.0,
    step=0.1
)

st.divider()

if st.button("Predict Rain Probability"):

    # Validate weather inputs
    if (
        humidity < 0 or humidity > 100
        or cloud_cover < 0 or cloud_cover > 100
        or temperature < -10 or temperature > 50
        or wind_speed < 0 or wind_speed > 100
        or pressure < 950 or pressure > 1100
    ):
        st.error("Please enter valid weather values.")
        st.stop()

    # Get current time features
    now = datetime.now()

    hour = now.hour
    day_of_week = now.weekday()
    month = now.month

    # Create model input
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
    rain_probability = model.predict_proba(
        input_data
    )[0][1]

    # Convert to percentage
    rain_probability_percentage = rain_probability * 100

    # Interpret probability
    if rain_probability_percentage < 30:
        chance = "Low"
    elif rain_probability_percentage < 60:
        chance = "Moderate"
    else:
        chance = "High"

    # Display result
    st.subheader("Rain Prediction")

    st.metric(
        label="Rain Probability",
        value=f"{rain_probability_percentage:.2f}%"
    )

    if chance == "Low":
        st.info("🌤️ Low chance of rain in the next hour.")

    elif chance == "Moderate":
        st.warning("⛅ Moderate chance of rain in the next hour.")

    else:
        st.error("🌧️ High chance of rain in the next hour.")
