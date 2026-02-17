import hopsworks
import joblib
import os
import pandas as pd
import shap
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# ---------------------------------------------------------
# 1. Setup & Connection
# ---------------------------------------------------------
print("🔌 Connecting to Hopsworks...")
load_dotenv()

try:
    # API Key check
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        print("❌ Error: HOPSWORKS_API_KEY not found in .env file")
        exit(1)

    project = hopsworks.login(project="aqi_predictor10", api_key_value=api_key)
    fs = project.get_feature_store()

except Exception as e:
    print(f"❌ Connection Failed: {e}")
    exit(1)

# ---------------------------------------------------------
# 2. Load Model (Local File: best_model.pkl) 📂
# ---------------------------------------------------------
print("📥 Loading Model...")

# Tumne kaha tha file ka naam 'best_model.pkl' kar diya hai locally
model_filename = "best_model.pkl" 

if os.path.exists(model_filename):
    model = joblib.load(model_filename)
    print(f"✅ Local Model Loaded: {model_filename}")
elif os.path.exists(f"models/{model_filename}"):
    model = joblib.load(f"models/{model_filename}")
    print(f"✅ Local Model Loaded: models/{model_filename}")
else:
    print(f"❌ Error: '{model_filename}' nahi mili! Make sure file isi folder mein hai.")
    exit(1)

# ---------------------------------------------------------
# 3. Fetch Data (Correct Feature Group Name & Version) 📊
# ---------------------------------------------------------
print("📊 Fetching Data for Analysis...")

try:
    # --- YAHAN CHANGE KIYA HAI ---
    # Tumhari file ke mutabiq naam 'aqi_features_main' aur version 2 hai
    fg_name = "aqi_features_main"
    fg_version = 2
    
    print(f"   Downloading from Feature Group: {fg_name} (v{fg_version})...")
    fg = fs.get_feature_group(name=fg_name, version=fg_version)
    df = fg.read()
    
    # --- Data Cleaning ---
    # Training file mein tumne sirf 'date' drop kiya tha inputs ke liye
    # Target column (us_aqi) ko bhi drop karna padega SHAP ke liye
    
    cols_to_drop = ['date', 'us_aqi'] 
    
    # Safe drop (errors='ignore' taake agar column na ho to crash na ho)
    X_sample = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Sirf numeric columns rakho (String/Object se SHAP crash ho jata hai)
    X_sample = X_sample.select_dtypes(include=['number'])
    
    # Sample 100 rows (SHAP slow hota hai)
    if len(X_sample) > 100:
        X_sample = X_sample.sample(100, random_state=42)
    
    print(f"✅ Data Ready! Shape: {X_sample.shape}")
    print(f"   Features: {list(X_sample.columns)}")

except Exception as e:
    print(f"❌ Data Fetch Failed: {e}")
    exit(1)

# ---------------------------------------------------------
# 4. SHAP Analysis & Plotting 📈
# ---------------------------------------------------------
print("🧠 Calculating SHAP Values...")

try:
    # --- YAHAN FIX KIYA HAI ---
    # VotingRegressor ke liye 'TreeExplainer' nahi chalta.
    # Hum generic 'Explainer' use karenge jo model.predict function ko use karega.
    
    # Ek chota background dataset banate hain (Speed badhane ke liye)
    # Ye batata hai ke "Average" values kya hain
    background_data = X_sample.median().values.reshape(1, -1)
    
    # Generic Explainer setup
    # Note: Hum model object nahi, balke model.predict function pass kar rahe hain
    explainer = shap.KernelExplainer(model.predict, X_sample)
    
    print("   ⏳ This might take 1-2 minutes for VotingRegressor. Please wait...")
    shap_values = explainer.shap_values(X_sample)

    # Plots Folder Create
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # --- GRAPH 1: Summary Plot ---
    print("🔹 Generating Summary Plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("What affects Air Quality the most?")
    plt.savefig("plots/shap_summary.png", bbox_inches='tight')
    plt.close()
    print("   👉 Saved: plots/shap_summary.png")

    # --- GRAPH 2: Bar Plot ---
    print("🔹 Generating Bar Plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title("Top Feature Importance")
    plt.savefig("plots/shap_bar.png", bbox_inches='tight')
    plt.close()
    print("   👉 Saved: plots/shap_bar.png")

    print("\n🎉 Success! Images saved in 'plots' folder.")

except Exception as e:
    print(f"⚠️ SHAP Calculation Error: {e}")