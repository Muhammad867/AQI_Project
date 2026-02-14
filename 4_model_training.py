import hopsworks
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
import joblib
import os
from dotenv import load_dotenv

# --- LOAD SECRETS ---
load_dotenv()

# --- CONFIGURATION ---
HOPSWORKS_PROJECT = "aqi_predictor10"
FEATURE_VIEW_NAME = "aqi_view_main"
FEATURE_VIEW_VERSION = 2
# Hum model ka naam generic rakhenge taake jo bhi jeete, isi naam se save ho
MODEL_NAME = "aqi_best_model" 

print("🚀 Connecting to Hopsworks...")
try:
    project = hopsworks.login(project=HOPSWORKS_PROJECT)
    fs = project.get_feature_store()
    mr = project.get_model_registry()
except Exception as e:
    print(f"❌ Login Error: {e}")
    exit()

# 1. GET FEATURE VIEW
try:
    feature_view = fs.get_feature_view(name=FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)
    print("✅ Feature View Found.")
except:
    print("❌ Feature View nahi mila. Backfill run karein.")
    exit()


# --- CLEANUP: DELETE OLD TRAINING DATASETS (FIXED) ---
print("🧹 Cleaning up old training datasets...")
try:
    # 1. Saare purane datasets ki list mangwao
    training_datasets = feature_view.get_training_datasets()
    
    for td in training_datasets:
        # 2. Version number nikalo
        v = td.version
        
        # 3. Feature View ko bolo ke is version ko uda de
        print(f"   🗑️ Deleting Training Data Version: {v}")
        feature_view.delete_training_dataset(v)
        
    print("✨ Cleanup Complete. Storage is free!")
except Exception as e:
    print(f"⚠️ Cleanup failed: {e}")

# 2. CREATE TRAINING DATA
print("📊 Creating Training Data...")
X_train, X_test, y_train, y_test = feature_view.train_test_split(
    test_size=0.2, 
    description="Model Comparison Split",
    data_format="pandas"
)

# Cleanup
X_train = X_train.drop(columns=["date"])
X_test = X_test.drop(columns=["date"])
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

print(f"✅ Data Ready! Train Shape: {X_train.shape}")

# 3. DEFINE CANDIDATES (PLAYERS) 🤖
print("🤖 Defining 5 Candidates...")

# Individual Models
rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
lgbm = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=10, random_state=42, verbose=-1)
lr = LinearRegression()

# Ensemble (Sabka mix)
voting = VotingRegressor(
    estimators=[('xgb', xgb_model), ('lgb', lgbm), ('rf', rf)]
)

# List of models to compete
candidates = [
    ("RandomForest", rf),
    ("XGBoost", xgb_model),
    ("LightGBM", lgbm),
    ("LinearRegression", lr),
    ("VotingEnsemble", voting)
]

# 4. START THE RACE 🏎️
print("\n🏎️ Starting Race (Training & Evaluating)...")

best_model = None
best_rmse = float('inf') # Shuru mein infinity maante hain
best_metrics = {}
best_name = ""

for name, model in candidates:
    print(f"   🏃 Training {name}...", end=" ")
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # Calculate RMSE
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"-> RMSE: {rmse:.4f}")
    
    # Compare: Kya ye pichle best se acha hai?
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model
        best_name = name
        best_metrics = {"rmse": rmse, "mae": mae, "r2": r2}

print(f"\n🏆 WINNER: {best_name} with RMSE: {best_rmse:.4f}")

# 5. SAVE ONLY THE WINNER 💾
print(f"💾 Saving Best Model ({best_name}) to Hopsworks...")

os.makedirs("models", exist_ok=True)
model_filename = "models/best_model.pkl"

# --- REPLACEMENT LOGIC ---
try:
    print(f"   🔍 Checking for existing model...")
    old_model = mr.get_model(MODEL_NAME, version=None)
    old_model.delete()
    print(f"   🗑️ Deleted old version.")
except:
    pass
# -------------------------
# --- COMPRESS HERE (Important!) ---
# compress=3 se file size ~30MB ho jayega. Upload/Download fast hoga.
joblib.dump(best_model, model_filename, compress=3)

input_schema = Schema(X_train)
output_schema = Schema(pd.DataFrame({'us_aqi': [10.5]}))
model_schema = ModelSchema(input_schema, output_schema)

# Upload Winner
aqi_model = mr.python.create_model(
    name=MODEL_NAME,
    metrics=best_metrics,
    model_schema=model_schema,
    input_example=X_train.sample(),
    description=f"Best Performing Model: {best_name}" # Description mein likha hoga kaun jeeta
)

aqi_model.save(model_filename)

print(f"🎉 Success! {best_name} is LIVE as '{MODEL_NAME}'.")