# 🌧️ Weather Prediction System

An ML-based rainfall prediction system that predicts the **probability of rainfall in Indore during the next hour** using current weather conditions.

## 📌 Project Overview

The system uses historical weather data from Indore to train a machine learning model.

The model takes current weather conditions such as:

- Temperature
- Relative Humidity
- Wind Speed
- Cloud Cover
- Atmospheric Pressure
- Hour
- Day of Week
- Month

and predicts the probability of rainfall during the **next hour**.

## 🎯 Objective

The main objective of this project is to build a simple machine learning system that provides an understandable rainfall probability instead of only predicting `Rain` or `No Rain`.

Example:

```text
Rain Probability: 87.50%
Chance: High
🧠 Machine Learning Approach
Model

Random Forest Classifier

The model uses:

150 decision trees
Maximum tree depth: 12
Minimum samples per leaf: 3
Balanced class weights
Fixed random state for reproducibility
Probability Calibration

The Random Forest probabilities are calibrated using:

Sigmoid Calibration

Calibration improves the reliability of predicted probabilities.

📊 Dataset

The dataset contains hourly weather observations for Indore.

Features Used
Feature	Description
temperature_2m	Temperature in °C
relative_humidity_2m	Relative humidity in %
wind_speed_10m	Wind speed
cloud_cover	Cloud cover in %
pressure_msl	Atmospheric pressure
hour	Hour of the day
day_of_week	Day of the week
month	Month
Target

The target variable is:

rain_next_hour

It is created from the precipitation recorded in the following hour.

precipitation > 0 → Rain
precipitation = 0 → No Rain

precipitation itself is not used as a model input to avoid target leakage.

🔄 Project Workflow
Raw Weather Data
       ↓
Data Cleaning
       ↓
Duplicate Removal
       ↓
Hourly Timeline Validation
       ↓
Next-Hour Target Creation
       ↓
Feature Engineering
       ↓
Exploratory Data Analysis
       ↓
Chronological Train/Test Split
       ↓
Random Forest Training
       ↓
Model Evaluation
       ↓
Probability Calibration
       ↓
Model Saving
       ↓
Prediction Pipeline
       ↓
Streamlit Web Interface
📈 Model Performance

The final model was evaluated on an untouched chronological test set.

Metric	Score
Accuracy	63.12%
Precision	56.27%
Recall	88.35%
F1 Score	68.76%
ROC-AUC	72.51%
Original Brier Score	0.2364
Calibrated Brier Score	0.2247

The calibrated model achieved a lower Brier Score, indicating improved probability calibration.

🖥️ Web Interface

The project includes a Streamlit web application where users can enter:

Temperature
Relative Humidity
Wind Speed
Cloud Cover
Pressure

The application then displays:

Rain Probability: XX.XX%

Chance: Low / Moderate / High
📁 Project Structure
weather_prediction_system/
│
├── data/
│   ├── indore_weather_data.csv
│   └── Figure_1.png
│
├── models/
│   ├── indore_rainfall_model.joblib
│   └── model_info.joblib
│
├── prepare_v2.py
├── predict.py
├── app.py
├── .gitignore
└── README.md
⚙️ Installation

Clone the repository:

git clone https://github.com/raghavin82/weather_prediction_system.git

Move into the project directory:

cd weather_prediction_system

Create a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate

Install the required libraries:

pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit
▶️ Run the Prediction Program

Run:

python predict.py

Enter the current weather conditions when prompted.

🌐 Run the Streamlit Application

Run:

streamlit run app.py

The application will open in your browser.

🧪 Testing

The system was tested using:

Normal weather conditions
Rainy weather conditions
Dry weather conditions
Minimum and maximum input boundaries
Repeated identical inputs
Saved model loading
End-to-end prediction through Streamlit

Example rainy-condition test:

Temperature: 26°C
Humidity: 90%
Wind Speed: 15 km/h
Cloud Cover: 100%
Pressure: 1000 hPa

Rain Probability: 87.50%
Chance: High
⚠️ Limitations
The model is trained specifically on Indore weather data.
Predictions depend on the quality and coverage of the historical dataset.
The model predicts the probability of rainfall during the next hour, not long-term weather conditions.
Model performance can vary with seasonal weather patterns.
The system is an academic ML project and should not be treated as an official weather forecasting service.
🚀 Future Improvements

Possible future improvements include:

Adding more historical weather data
Using additional meteorological features
Comparing multiple ML algorithms
Hyperparameter optimization
Improved probability calibration
Real-time weather API integration
Deployment on a cloud platform
Adding prediction history and visualizations
👨‍💻 Author

Raghav Patel

Computer Science & Engineering (AIML)

📜 License

This project is developed for educational and academic purposes.