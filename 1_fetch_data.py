import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime

# --- CONFIGURATION ---
CITY_LAT = 24.8607  # Karachi
CITY_LON = 67.0011
START_DATE = "2025-01-01"
# Important: End Date aaj ki honi chahiye
END_DATE = datetime.now().strftime("%Y-%m-%d")

print(f"📥 Fetching data from {START_DATE} to {END_DATE}...")

# 1. SETUP API CLIENT
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# 2. DEFINE PARAMETERS
# Weather (Mausam)
weather_cols = ["temperature_2m", "relative_humidity_2m", "rain", "wind_speed_10m", "surface_pressure", "cloud_cover"]
params_w = {
    "latitude": CITY_LAT, "longitude": CITY_LON,
    "start_date": START_DATE, "end_date": END_DATE,
    "hourly": weather_cols
}

# Air Quality (Hawa)
air_cols = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "us_aqi"]
params_a = {
    "latitude": CITY_LAT, "longitude": CITY_LON,
    "start_date": START_DATE, "end_date": END_DATE,
    "hourly": air_cols
}

# 3. CALL API
print("☁️ Calling Weather API...")
weather_res = openmeteo.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params_w)[0]

print("💨 Calling Air Quality API...")
air_res = openmeteo.weather_api("https://air-quality-api.open-meteo.com/v1/air-quality", params=params_a)[0]

# 4. PROCESS DATA
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

print("⚙️ Processing Dataframes...")
df_w = to_df(weather_res, weather_cols)
df_a = to_df(air_res, air_cols)

# Merge
df = pd.merge(df_w, df_a, on="date")

# Rename columns (remove _2m, _10m etc)
df.columns = [c.replace("_2m", "").replace("_10m", "") for c in df.columns]

# 5. SAVE
filename = "aqi_data_for_eda.csv"
df.to_csv(filename, index=False)
print(f"✅ Success! Updated data saved to '{filename}' (Rows: {len(df)})")