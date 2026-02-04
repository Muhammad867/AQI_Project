import streamlit as st
import hopsworks
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

# --- PAGE CONFIG ---
st.set_page_config(page_title="AQI Forecast (3-Day)", page_icon="😷", layout="centered")

# --- CUSTOM CSS (Styling for Cards) ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 10px;
    }
    .aqi-val { font-size: 36px; font-weight: bold; }
    .aqi-label { font-size: 20px; font-weight: bold; padding: 5px; border-radius: 5px; color: white;}
    </style>
""", unsafe_allow_html=True)

# --- HELPER: AQI CATEGORY & COLOR ---
def get_aqi_category(aqi):
    if aqi <= 50: return "Good", "#00e400"  # Green
    elif aqi <= 100: return "Moderate", "#ffff00"  # Yellow (Text needs to be black usually)
    elif aqi <= 150: return "Unhealthy for Sensitive Groups", "#ff7e00"  # Orange
    elif aqi <= 200: return "Unhealthy", "#ff0000"  # Red
    elif aqi <= 300: return "Very Unhealthy", "#8f3f97"  # Purple
    else: return "Hazardous", "#7e0023"  # Maroon

# --- 1. CONNECT & FETCH DATA ---
@st.cache_resource
def get_hopsworks_data():
    load_dotenv()
    try:
        project = hopsworks.login(project="aqi_predictor10")
        fs = project.get_feature_store()
        mr = project.get_model_registry()
        
        # A. Fetch Model Metrics
        model = mr.get_model("aqi_best_model", version=None)
        # metrics = model.metrics # {rmse: 8.5, mae: 5.2, ...}
        metrics = model.training_metrics
        
        # B. Fetch Predictions
        fg = fs.get_feature_group(name="aqi_predictions_daily", version=1)
        df = fg.read()
        return df, metrics
    except Exception as e:
        st.error(f"Error connecting to Hopsworks: {e}")
        return None, None

# --- APP LAYOUT ---
st.title("🌫️ Karachi 3-Day AQI Forecast")

with st.spinner("Fetching latest data from Feature Store..."):
    raw_df, metrics = get_hopsworks_data()

if raw_df is not None and metrics is not None:

    # --- SECTION 1: MODEL METRICS ---
    st.subheader("📊 Model Performance")
    col1, col2, col3 = st.columns(3)
    
    # Metrics display kar rahe hain
    col1.metric("RMSE (Error)", f"{metrics.get('rmse', 0):.2f}")
    col2.metric("MAE (Avg Diff)", f"{metrics.get('mae', 0):.2f}")
    col3.metric("R² Score (Accuracy)", f"{metrics.get('r2', 0):.2f}")
    
    st.divider()

    # --- SECTION 2: PROCESS DATA (HOURLY -> DAILY) ---
    # Date processing
    raw_df['date'] = pd.to_datetime(raw_df['date'], unit='ms')
    
    # Sirf Future dates rakhein (Today onwards)
    today = pd.Timestamp.now().normalize()
    future_df = raw_df[raw_df['date'] >= today].copy()
    
    # Extract only the Date (remove time)
    future_df['day_date'] = future_df['date'].dt.date
    
    # GROUP BY DAY (Average nikalne ke liye)
    daily_df = future_df.groupby('day_date')['predicted_aqi'].mean().reset_index()
    
    # Sort and take first 3 days
    daily_df = daily_df.sort_values('day_date').head(3)

    # --- SECTION 3: DISPLAY CARDS ---
    st.subheader("📅 Next 3 Days Forecast")

    if not daily_df.empty:
        # Create 3 Columns dynamically
        cols = st.columns(len(daily_df))
        
        for index, (i, row) in enumerate(daily_df.iterrows()):
            date_str = row['day_date'].strftime("%A, %d %b") # e.g. Monday, 31 Jan
            aqi_val = int(row['predicted_aqi'])
            label, color = get_aqi_category(aqi_val)
            
            # Text color fix for Yellow background
            text_color = "black" if color == "#ffff00" else "white"

            with cols[index]:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{date_str}</h4>
                    <div class="aqi-val">{aqi_val}</div>
                    <div style="font-size: 14px; margin-bottom: 10px;">Overall AQI Forecast</div>
                    <div class="aqi-label" style="background-color: {color}; color: {text_color};">
                        {label}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No future predictions found. Please run the batch script.")