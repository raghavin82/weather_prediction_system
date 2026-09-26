import pandas as pd


# ============================================================
# Phase 1 — Project Setup
# ============================================================

print("\n========== PHASE 1: PROJECT SETUP ==========")

# Load dataset
df = pd.read_csv("data/indore_weather_data.csv")

print("Dataset loaded successfully.")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# Phase 2 — Understand the Raw Dataset
# ============================================================

print("\n========== PHASE 2: DATASET INSPECTION ==========")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nBasic statistics:")
print(df.describe())


# ============================================================
# Phase 3 — Clean and Prepare Dataset
# ============================================================

print("\n========== PHASE 3: DATA CLEANING ==========")


# Convert time column
df["time"] = pd.to_datetime(df["time"])

# Sort chronologically
df = df.sort_values("time").reset_index(drop=True)

print("\nTime range:")
print("Start:", df["time"].min())
print("End:", df["time"].max())


# Check raw duplicates
print("\nExact duplicate rows:", df.duplicated().sum())
print("Duplicate timestamps:", df["time"].duplicated().sum())


# Check time differences
time_diff = df["time"].diff().dropna()

print("\nTime difference counts:")
print(time_diff.value_counts().head(10))


# Remove exact duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

print("\nAfter removing exact duplicate rows:")
print("Shape:", df.shape)
print("Duplicate timestamps:", df["time"].duplicated().sum())


# Check remaining duplicate timestamps
duplicate_times = df[df["time"].duplicated(keep=False)]

print("\nRows with duplicate timestamps:")
print(duplicate_times.head(20))

print(
    "\nNumber of rows with duplicate timestamps:",
    len(duplicate_times)
)


# Combine duplicate timestamps
# First non-missing value is retained for each column
df = (
    df.groupby("time", as_index=False)
      .first()
)

print("\nAfter combining duplicate timestamps:")
print("Shape:", df.shape)
print("Duplicate timestamps:", df["time"].duplicated().sum())


# Validate hourly timeline
time_diff = df["time"].diff().dropna()

print("\nTime difference counts after combining timestamps:")
print(time_diff.value_counts().head(10))

print("\nLargest time gap:")
print(time_diff.max())


# ============================================================
# Phase 4 — Create Next-Hour Target
# ============================================================

print("\n========== PHASE 4: NEXT-HOUR TARGET ==========")


# Get precipitation from the next hour
df["next_hour_precipitation"] = (
    df["precipitation"].shift(-1)
)


# Final row has no next-hour observation
df = df.dropna(
    subset=["next_hour_precipitation"]
).copy()


# Create binary target
# 0 = no rain in next hour
# 1 = rain in next hour
df["rain_next_hour"] = (
    df["next_hour_precipitation"] > 0
).astype(int)


print("\nTarget distribution:")
print(df["rain_next_hour"].value_counts())

print("\nTarget percentages:")
print(
    df["rain_next_hour"]
      .value_counts(normalize=True) * 100
)

print("\nShape after creating target:")
print(df.shape)


# ============================================================
# Phase 5 — Feature Engineering
# ============================================================

print("\n========== PHASE 5: FEATURE ENGINEERING ==========")


# Time-based features
df["hour"] = df["time"].dt.hour
df["day_of_week"] = df["time"].dt.dayofweek
df["month"] = df["time"].dt.month


print("\nTime features:")
print(
    df[
        ["time", "hour", "day_of_week", "month"]
    ].head()
)


# Check missing weather features
weather_columns = [
    "wind_speed_10m",
    "cloud_cover",
    "pressure_msl"
]

missing_weather = df[
    df[weather_columns].isnull().any(axis=1)
]

print("\nRows with missing weather features:")
print(
    missing_weather[
        [
            "time",
            "wind_speed_10m",
            "cloud_cover",
            "pressure_msl"
        ]
    ]
)


# Remove rows with missing model features
df = df.dropna(
    subset=weather_columns
).copy()


print("\nAfter removing missing weather rows:")
print("Shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())


# Validate timeline after cleaning
time_diff = df["time"].diff().dropna()

print("\nTime difference after cleaning:")
print(time_diff.value_counts().head(10))

print("\nLargest time gap:")
print(time_diff.max())


# Define model features
features = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "cloud_cover",
    "pressure_msl",
    "hour",
    "day_of_week",
    "month"
]


print("\nSelected model features:")
print(features)


# Create X and y
X = df[features]
y = df["rain_next_hour"]


print("\nFeature matrix shape:", X.shape)
print("Target shape:", y.shape)


# Final validation
print("\nFinal duplicate timestamps:")
print(df["time"].duplicated().sum())

print("\nFinal target distribution:")
print(y.value_counts())

print("\nFinal target percentage:")
print(
    y.value_counts(normalize=True) * 100
)

print("\nFinal feature columns:")
print(X.columns.tolist())


# ============================================================
# Phase 6 — Exploratory Data Analysis
# ============================================================

print("\n========== PHASE 6: EXPLORATORY DATA ANALYSIS ==========")


# ------------------------------------------------------------
# Rain vs No-Rain Distribution
# ------------------------------------------------------------

print("\n--- Rain vs No-Rain Distribution ---")

print("\nRain counts:")
print(y.value_counts())

print("\nRain percentages:")
print(
    y.value_counts(normalize=True) * 100
)


# ------------------------------------------------------------
# Rain Probability by Month
# ------------------------------------------------------------

print("\n--- Rain Probability by Month ---")

monthly_rain = (
    df.groupby("month")["rain_next_hour"]
      .mean() * 100
)

print(monthly_rain)


# ------------------------------------------------------------
# Rain Probability by Hour
# ------------------------------------------------------------

print("\n--- Rain Probability by Hour ---")

hourly_rain = (
    df.groupby("hour")["rain_next_hour"]
      .mean() * 100
)

print(hourly_rain)


# ------------------------------------------------------------
# Humidity vs Rain
# ------------------------------------------------------------

print("\n--- Average Humidity ---")

humidity_rain = (
    df.groupby("rain_next_hour")["relative_humidity_2m"]
      .mean()
)

print(humidity_rain)


# ------------------------------------------------------------
# Cloud Cover vs Rain
# ------------------------------------------------------------

print("\n--- Average Cloud Cover ---")

cloud_rain = (
    df.groupby("rain_next_hour")["cloud_cover"]
      .mean()
)

print(cloud_rain)


# ------------------------------------------------------------
# Pressure vs Rain
# ------------------------------------------------------------

print("\n--- Average Pressure ---")

pressure_rain = (
    df.groupby("rain_next_hour")["pressure_msl"]
      .mean()
)

print(pressure_rain)


# ------------------------------------------------------------
# Temperature vs Rain
# ------------------------------------------------------------

print("\n--- Average Temperature ---")

temperature_rain = (
    df.groupby("rain_next_hour")["temperature_2m"]
      .mean()
)

print(temperature_rain)


# ------------------------------------------------------------
# Wind Speed vs Rain
# ------------------------------------------------------------

print("\n--- Average Wind Speed ---")

wind_rain = (
    df.groupby("rain_next_hour")["wind_speed_10m"]
      .mean()
)

print(wind_rain)


# ------------------------------------------------------------
# Correlation Analysis
# ------------------------------------------------------------

print("\n--- Correlation Matrix ---")

correlation_matrix = df[
    features + ["rain_next_hour"]
].corr()

print(correlation_matrix)

print("\nCorrelation with rain_next_hour:")

print(
    correlation_matrix["rain_next_hour"]
    .sort_values(ascending=False)
)


# ------------------------------------------------------------
# Optional Correlation Heatmap
# ------------------------------------------------------------

# Set this to True only when you want to generate the heatmap.
SHOW_HEATMAP = False


if SHOW_HEATMAP:

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()

    plt.show()

    plt.close()


# ------------------------------------------------------------
# EDA Conclusions
# ------------------------------------------------------------

print("\n--- EDA Conclusions ---")

print("""
1. Rain occurs less frequently than no-rain conditions.

2. Rain probability varies strongly by month, with higher
   values during the monsoon period.

3. Rain probability varies by hour, with higher values mainly
   during the morning to afternoon period.

4. Rainy next hours have substantially higher average humidity.

5. Rainy next hours have substantially higher average cloud cover.

6. Rainy next hours are associated with lower average pressure.

7. Wind speed is somewhat higher before rainy next hours.

8. Temperature shows only a small difference between rain
   and no-rain cases.

9. Cloud cover, humidity, and pressure show the strongest
   individual relationships with the rain target.

10. Correlation shows association, not causation, and does not
    represent final model feature importance.
""")


