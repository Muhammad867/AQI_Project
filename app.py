import streamlit as st
import hopsworks
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

# --- PAGE CONFIG ---
st.set_page_config(page_title="AQI Forecast (3-Day)", page_icon="😷", layout="centered")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .aqi-val { font-size: 36px; font-weight: bold; }
    .aqi-label { font-size: 20px; font-weight: bold; padding: 5px; border-radius: 5px; color: white;}
    </style>
""", unsafe_allow_html=True)

# --- HELPER: AQI CATEGORY ---
def get_aqi_category(aqi):
    if aqi <= 50: return "Good", "#00e400"
    elif aqi <= 100: return "Moderate", "#ffff00"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups", "#ff7e00"
    elif aqi <= 200: return "Unhealthy", "#ff0000"
    elif aqi <= 300: return "Very Unhealthy", "#8f3f97"
    else: return "Hazardous", "#7e0023"

# --- 1. CONNECT & FETCH DATA ---
@st.cache_resource
def get_hopsworks_data():
    load_dotenv()
    try:
        try:
            api_key = st.secrets["HOPSWORKS_API_KEY"]
        except:
            api_key = os.getenv("HOPSWORKS_API_KEY")

        project = hopsworks.login(project="aqi_predictor10", api_key_value=api_key)
        fs = project.get_feature_store()
        mr = project.get_model_registry()
        
        # A. Fetch Model Metrics & Name
        try:
            model = mr.get_model("aqi_best_model", version=None)
            metrics = model.training_metrics
            # ✨ NEW: Get Model Name
            model_name = model.description if model.description else "XGBoost Regressor"
        except:
            metrics = {}
            model_name = "Unknown Model"
        
        # B. Fetch Predictions
        fg = fs.get_feature_group(name="aqi_predictions_daily", version=1)
        df = fg.read()
        
        return df, metrics, model_name  # ✨ Returning 3 items now
        
    except Exception as e:
        st.error(f"Error connecting to Hopsworks: {e}")
        return None, None, None

# --- APP LAYOUT ---
st.title("🌫️ Karachi 3-Day AQI Forecast")

with st.spinner("Fetching latest data from Feature Store..."):
    # ✨ Unpacking 3 values
    raw_df, metrics, model_name = get_hopsworks_data()

if raw_df is not None:

    # --- SECTION 0: ACTIVE MODEL NAME (NEW) ---
    st.info(f"🤖 **Active Model:** {model_name}") 

    # --- SECTION 1: MODEL METRICS ---
    if metrics:
        st.subheader("📊 Model Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE (Error)", f"{metrics.get('rmse', 0):.2f}")
        col2.metric("MAE (Avg Diff)", f"{metrics.get('mae', 0):.2f}")
        col3.metric("R² Score (Accuracy)", f"{metrics.get('r2', 0):.2f}")
        st.divider()

    # --- SECTION 2: PROCESS DATA ---
    if raw_df['date'].dtype == 'object':
         raw_df['date'] = pd.to_datetime(raw_df['date'])
    else:
         raw_df['date'] = pd.to_datetime(raw_df['date'], unit='ms')
    
    today = pd.Timestamp.now().normalize()
    future_df = raw_df[raw_df['date'] >= today].copy()
    future_df['day_date'] = future_df['date'].dt.date
    
    col_name = 'predicted_aqi' if 'predicted_aqi' in future_df.columns else 'predicted_pm2_5'
    
    daily_df = future_df.groupby('day_date')[col_name].mean().reset_index()
    daily_df.rename(columns={col_name: 'predicted_aqi'}, inplace=True)
    
    daily_df = daily_df.sort_values('day_date').head(3)

    # --- SECTION 3: DISPLAY CARDS ---
    st.subheader("📅 Next 3 Days Forecast")

    if not daily_df.empty:
        cols = st.columns(len(daily_df))
        
        for index, (i, row) in enumerate(daily_df.iterrows()):
            date_str = row['day_date'].strftime("%A, %d %b")
            aqi_val = int(row['predicted_aqi'])
            label, color = get_aqi_category(aqi_val)
            text_color = "black" if color == "#ffff00" else "white"

            with cols[index]:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{date_str}</h4>
                    <div class="aqi-val">{aqi_val}</div>
                    <div style="font-size: 14px; margin-bottom: 10px;">Average AQI</div>
                    <div class="aqi-label" style="background-color: {color}; color: {text_color};">
                        {label}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # --- SECTION 4: PLOT PREDICTION GRAPH ---
        st.divider()
        st.subheader("📈 AQI Trend (Next 3 Days)")
        
        chart_data = daily_df.set_index('day_date')[['predicted_aqi']]
        chart_data.columns = ['Forecasted AQI']
        st.line_chart(chart_data, color="#FF4B4B")
        
    else:
        st.warning("⚠️ No future predictions found. Please run the batch script or check dates.")