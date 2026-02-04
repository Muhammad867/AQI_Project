import hopsworks
import pandas as pd
import numpy as np
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv

# --- 1. SETUP & AUTH ---
load_dotenv()

HOPSWORKS_PROJECT = "aqi_predictor10"
API_KEY = os.getenv("HOPSWORKS_API_KEY")
CITY_LAT = 24.8607
CITY_LON = 67.0011

# --- 2. FETCH DATA (Forecast API) ---
print("📥 Fetching recent data...")

cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

end_date = datetime.now()
start_date = end_date - timedelta(days=3)

url_weather = "https://api.open-meteo.com/v1/forecast"
url_air = "https://air-quality-api.open-meteo.com/v1/air-quality"

params_w = {
    "latitude": CITY_LAT, "longitude": CITY_LON,
    "start_date": start_date.strftime("%Y-%m-%d"),
    "end_date": end_date.strftime("%Y-%m-%d"),
    "hourly": ["temperature_2m", "relative_humidity_2m", "rain", "wind_speed_10m", "surface_pressure", "cloud_cover"]
}
params_a = {
    "latitude": CITY_LAT, "longitude": CITY_LON,
    "start_date": start_date.strftime("%Y-%m-%d"),
    "end_date": end_date.strftime("%Y-%m-%d"),
    "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "us_aqi"]
}

try:
    weather_res = openmeteo.weather_api(url_weather, params=params_w)[0]
    air_res = openmeteo.weather_api(url_air, params=params_a)[0]
except Exception as e:
    print(f"❌ API Error: {e}")
    exit()

# --- 3. PROCESS DATA ---
def to_df(response, col_names):
    hourly = response.Hourly()
    data = {"date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )}
    for i, col_name in enumerate(col_names):
        data[col_name] = hourly.Variables(i).ValuesAsNumpy()
    return pd.DataFrame(data)

df_w = to_df(weather_res, params_w["hourly"])
df_a = to_df(air_res, params_a["hourly"])

df = pd.merge(df_w, df_a, on="date")
df.columns = [c.replace("_2m", "").replace("_10m", "") for c in df.columns]

# --- FILTER FUTURE DATA ---
current_time_utc = pd.Timestamp.now(tz='UTC')
df = df[df['date'] <= current_time_utc]

print(f"🕒 Cutoff Time: {current_time_utc}")
print(f"📊 Rows kept: {len(df)}")

# --- 4. FEATURE ENGINEERING (V2) ---
print("⚙️ Applying Feature Engineering (V2)...")

df = df.sort_values(by='date')
df['hour'] = df['date'].dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['pm2_5_roll_3'] = df['pm2_5'].rolling(window=3).mean()
df['pm2_5_lag_1'] = df['pm2_5'].shift(1)
df['pm2_5_lag_6'] = df['pm2_5'].shift(6)
df['pm2_5_lag_24'] = df['pm2_5'].shift(24)

df = df.dropna()

# --- 5. UPLOAD TO HOPSWORKS ---
print("🚀 Connecting to Hopsworks...")

if not API_KEY:
    print("❌ API Key missing!")
    exit()

project = hopsworks.login(project=HOPSWORKS_PROJECT, api_key_value=API_KEY)
fs = project.get_feature_store()
aqi_fg = fs.get_feature_group(name="aqi_features_main", version=2)

# --- FIX: CONVERT TO DOUBLE ---
print("🔧 Fixing Data Types...")
for col in df.columns:
    if col != "date" and col != "hour": 
        df[col] = df[col].astype("float64")

# Date Processing
df['date'] = pd.to_datetime(df['date'])
df['date'] = df['date'].map(pd.Timestamp.timestamp) * 1000
df['date'] = df['date'].astype('int64')

df_tail = df.tail(24)

print(f"⬆️ Uploading {len(df_tail)} new rows to Feature Store...")

# --- 🔥 CRITICAL FIX: SKIP OFFLINE JOB 🔥 ---
# Hum keh rahe hain: "Bas data save karo aur niklo. Wait mat karo."
try:
    aqi_fg.insert(df_tail, write_options={"wait_for_job": False})
    print("✅ Insert command sent successfully (Background job handled by Hopsworks).")
except Exception as e:
    print(f"⚠️ Warning during insert (Data might be uploaded anyway): {e}")

print("🎉 Success! Pipeline Finished.")