import hopsworks
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv  # New Import

# --- LOAD SECRETS ---
load_dotenv()  # Ye .env file parhay ga

# --- CONFIGURATION ---
HOPSWORKS_PROJECT = "aqi_predictor10"
API_KEY = os.getenv("HOPSWORKS_API_KEY") # Key variable mein aa gayi

if not API_KEY:
    print("❌ Error: .env file mein 'HOPSWORKS_API_KEY' nahi mili!")
    exit()

# 1. LOAD DATA
print("📂 Loading data...")
try:
    df = pd.read_csv("aqi_data_for_eda.csv")
except FileNotFoundError:
    print("❌ Error: 'aqi_data_for_eda.csv' nahi mili. Pehle 1_fetch_data.py chalao.")
    exit()

# 2. CLEANING
cols_to_drop = ['day_of_week', 'wind_direction', 'month', 'ozone']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# 3. ADVANCED FEATURE ENGINEERING (V2)
print("🛠️ Engineering New Features...")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date')

df['hour'] = df['date'].dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['pm2_5_roll_3'] = df['pm2_5'].rolling(window=3).mean()
df['pm2_5_lag_1'] = df['pm2_5'].shift(1)
df['pm2_5_lag_6'] = df['pm2_5'].shift(6)
df['pm2_5_lag_24'] = df['pm2_5'].shift(24)
df = df.dropna()

print(f"✅ Final Features: {list(df.columns)}")

# 4. PREPARE FOR UPLOAD
df['date'] = df['date'].map(pd.Timestamp.timestamp) * 1000
df['date'] = df['date'].astype('int64')

# 5. CONNECT TO HOPSWORKS (SECURE LOGIN) 🔐
print("🚀 Connecting to Hopsworks...")

# Yahan hum 'api_key_value' pass kar rahe hain
project = hopsworks.login(
    project=HOPSWORKS_PROJECT, 
    api_key_value=API_KEY
)

fs = project.get_feature_store()

# 6. CREATE FEATURE GROUP (VERSION 2)
aqi_fg = fs.get_or_create_feature_group(
    name="aqi_features_main",
    version=2,
    primary_key=["date"], 
    event_time="date",
    description="AQI Data V2 with Rolling Means and Cyclical Time"
)

# 7. UPLOAD
print("⬆️ Uploading to Feature Store (Version 2)...")
aqi_fg.insert(df, write_options={"wait_for_job": False})

print("🎉 Success! Secure Login ke sath upload start ho gaya.")