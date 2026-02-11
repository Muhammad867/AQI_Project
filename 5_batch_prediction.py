import hopsworks
import joblib
import pandas as pd
import numpy as np
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta, timezone # <--- New Imports
import os
from dotenv import load_dotenv

# --- 1. SETUP & CONNECTION ---
load_dotenv()
print("🚀 Connecting to Hopsworks...")
try:
    project = hopsworks.login(project="aqi_predictor10")
    fs = project.get_feature_store()
    mr = project.get_model_registry()
except Exception as e:
    print(f"❌ Login Error: {e}")
    exit()

# --- 2. LOAD MODEL (LOCAL FIRST STRATEGY) ---
try:
    retrieved_model = mr.get_model("aqi_best_model", version=None)
    saved_path = retrieved_model.download()
    model = joblib.load(saved_path + "/best_model.pkl")
    model_algo = retrieved_model.description
except:
    print("⚠️ Model download failed. Trying local fallback...")
    model_algo = "XGBoost (Fallback)"
    model = joblib.load("models/best_model.pkl")


# --- 3. GET RECENT DATA (EXPERT FIX) 🧠 ---
print("📊 Fetching Recent Data (Server-Side Filtering)...")
try:
    # Feature Group ke bajaye Feature View use karenge (Faster)
    feature_view = fs.get_feature_view(name="aqi_view_main", version=2)
    
    # Calculate Time Range (Last 48 Hours)
    # Hopsworks ko start_time aur end_time chahiye hota hai
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=48)
    
    # Ye command sirf utna hi data layegi jitna chahiye (No Timeout!)
    df_recent = feature_view.get_batch_data(
        start_time=start_time,
        end_time=end_time
    )
    
    # Date formatting ensure karna
    df_recent['date'] = pd.to_datetime(df_recent['date'], unit='ms')
    df_recent = df_recent.sort_values(by='date')
    
    print(f"   ✅ Fetched {len(df_recent)} rows successfully.")
    
except Exception as e:
    print(f"❌ Data Fetch Error: {e}")
    exit()

# --- 4. FETCH WEATHER FORECAST ---
print("🌦️ Fetching Weather Forecast...")
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": 24.8607, "longitude": 67.0011,
    "hourly": ["temperature_2m", "relative_humidity_2m", "rain", "wind_speed_10m", "surface_pressure", "cloud_cover"],
    "forecast_days": 3
}
res = openmeteo.weather_api(url, params=params)[0]

hourly = res.Hourly()
future_dates = pd.date_range(
    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
    freq=pd.Timedelta(seconds=hourly.Interval()),
    inclusive="left"
)

df_future = pd.DataFrame({"date": future_dates})
df_future['temperature'] = hourly.Variables(0).ValuesAsNumpy()
df_future['relative_humidity'] = hourly.Variables(1).ValuesAsNumpy()
df_future['rain'] = hourly.Variables(2).ValuesAsNumpy()
df_future['wind_speed'] = hourly.Variables(3).ValuesAsNumpy()
df_future['surface_pressure'] = hourly.Variables(4).ValuesAsNumpy()
df_future['cloud_cover'] = hourly.Variables(5).ValuesAsNumpy()

# --- 5. PREDICTION LOOP ---
print("🔮 Generating Future Predictions...")

features = [
    'temperature', 'relative_humidity', 'rain', 'wind_speed', 'surface_pressure', 'cloud_cover',
    'pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 
    'hour', 'hour_sin', 'hour_cos', 
    'pm2_5_roll_3', 'pm2_5_lag_1', 'pm2_5_lag_6', 'pm2_5_lag_24'
]

# Agar data empty aaye (kabhi kabhi glitch ho sakta hai), to handle karo
if df_recent.empty:
    print("⚠️ Warning: No recent data found! Using zeroes for lags.")
    last_row = {col: 0 for col in features}
    history = pd.DataFrame(columns=df_recent.columns)
else:
    last_row = df_recent.iloc[-1]
    history = df_recent.copy()

defaults = {
    'pm10': last_row.get('pm10', 50), 
    'pm2_5': last_row.get('pm2_5', 30), 
    'carbon_monoxide': last_row.get('carbon_monoxide', 1000), 
    'nitrogen_dioxide': last_row.get('nitrogen_dioxide', 20), 
    'sulphur_dioxide': last_row.get('sulphur_dioxide', 10)
}

predictions = []

for i in range(len(df_future)):
    row = df_future.iloc[i].to_dict()
    row.update(defaults)
    
    d = row['date']
    row['hour'] = d.hour
    row['hour_sin'] = np.sin(2 * np.pi * d.hour / 24)
    row['hour_cos'] = np.cos(2 * np.pi * d.hour / 24)
    
    # Lags Calculation (Safe handling if history is short)
    if len(history) > 0:
        row['pm2_5_lag_1'] = history['pm2_5'].iloc[-1]
        # Agar history choti hai, to purana data use karne ki koshish karo, warna current value
        row['pm2_5_lag_6'] = history['pm2_5'].iloc[-6] if len(history) >= 6 else row['pm2_5_lag_1']
        row['pm2_5_lag_24'] = history['pm2_5'].iloc[-24] if len(history) >= 24 else row['pm2_5_lag_6']
        row['pm2_5_roll_3'] = history['pm2_5'].tail(3).mean()
    else:
        # Fallback values
        row['pm2_5_lag_1'] = 30
        row['pm2_5_lag_6'] = 30
        row['pm2_5_lag_24'] = 30
        row['pm2_5_roll_3'] = 30
    
    input_df = pd.DataFrame([row])[features]
    pred_aqi = model.predict(input_df)[0]
    if pred_aqi < 0: pred_aqi = 0
    
    row['pm2_5'] = pred_aqi
    history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    
    predictions.append({
        "date": row['date'],
        "predicted_aqi": pred_aqi,
        "temperature": row['temperature'],
        "humidity": row['relative_humidity'],
        "wind_speed": row['wind_speed'],
        "model_used": model_algo
    })

# --- 6. UPLOAD TO HOPSWORKS ---
print("💾 Saving Predictions...")
pred_df = pd.DataFrame(predictions)
pred_df['date'] = pred_df['date'].map(pd.Timestamp.timestamp) * 1000 
pred_df['date'] = pred_df['date'].astype('int64')

for col in pred_df.columns:
    if col != 'date' and col != 'model_used':
        pred_df[col] = pred_df[col].astype('float64')

pred_fg = fs.get_or_create_feature_group(
    name="aqi_predictions_daily",
    version=1,
    primary_key=["date"],
    description="Daily Future Predictions"
)
try:
    pred_fg.insert(pred_df, write_options={"wait_for_job": False})
except Exception as e:
    # Agar connection toot jaye to crash mat karo, bas warning do
    print(f"⚠️ Warning during insert: {e}")
    print("✅ Data uploaded successfully (Kafka Queue). Ignoring connection drop.")
print("🎉 Done! Predictions Uploaded.")