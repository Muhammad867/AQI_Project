# import hopsworks
# import pandas as pd
# import numpy as np  # Numpy zaroori hai square root ke liye
# import xgboost as xgb
# import lightgbm as lgb
# from sklearn.ensemble import RandomForestRegressor, VotingRegressor
# from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
# from hsml.schema import Schema
# from hsml.model_schema import ModelSchema
# import joblib
# import os
# from dotenv import load_dotenv

# # --- LOAD SECRETS ---
# load_dotenv()

# # --- CONFIGURATION ---
# HOPSWORKS_PROJECT = "aqi_predictor10"
# MODEL_NAME = "aqi_model_ensemble"
# FEATURE_VIEW_NAME = "aqi_view_main"
# FEATURE_VIEW_VERSION = 2

# print("🚀 Connecting to Hopsworks...")
# try:
#     project = hopsworks.login(project=HOPSWORKS_PROJECT)
#     fs = project.get_feature_store()
#     mr = project.get_model_registry()
# except Exception as e:
#     print(f"❌ Login Error: {e}")
#     exit()

# # 1. GET FEATURE GROUP
# try:
#     aqi_fg = fs.get_feature_group(name="aqi_features_main", version=2)
# except:
#     print("❌ Feature Group V2 nahi mila. Please run backfill script.")
#     exit()

# # 2. GET FEATURE VIEW (Ab ye stable hai)
# try:
#     feature_view = fs.get_feature_view(name=FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)
#     print("✅ Feature View Found.")
# except:
#     print("⚙️ Creating Feature View...")
#     feature_view = fs.create_feature_view(
#         name=FEATURE_VIEW_NAME,
#         version=FEATURE_VIEW_VERSION,
#         query=aqi_fg.select_all(),
#         labels=["us_aqi"]
#     )
#     print("✅ New Feature View Created.")

# # 3. CREATE TRAINING DATA
# print("📊 Creating Training Data...")
# X_train, X_test, y_train, y_test = feature_view.train_test_split(
#     test_size=0.2, 
#     description="Voting Ensemble Split",
#     data_format="pandas"
# )

# # Date drop karna
# X_train = X_train.drop(columns=["date"])
# X_test = X_test.drop(columns=["date"])

# # --- WARNING FIX: Y ko flatten karna ---
# y_train = y_train.values.ravel()
# y_test = y_test.values.ravel()

# print(f"✅ Data Ready! Train Shape: {X_train.shape}")

# # 4. TRAIN CHAMPION MODEL 🤖
# print("🤖 Training Voting Ensemble...")

# model_rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
# model_xgb = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
# model_lgb = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=10, random_state=42, verbose=-1)

# model = VotingRegressor(
#     estimators=[
#         ('xgb', model_xgb), 
#         ('lgb', model_lgb), 
#         ('rf', model_rf)
#     ]
# )

# model.fit(X_train, y_train)

# # 5. EVALUATE
# print("📉 Calculating Metrics...")
# preds = model.predict(X_test)

# mae = mean_absolute_error(y_test, preds)
# r2 = r2_score(y_test, preds)

# # --- RMSE FIX (Universal Method) ---
# mse = mean_squared_error(y_test, preds)
# rmse = np.sqrt(mse)  # Manual Square Root

# print(f"🎯 Final Score -> MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.3f}")

# # 6. SAVE TO REGISTRY
# print("💾 Saving Model to Hopsworks...")

# os.makedirs("models", exist_ok=True)
# joblib.dump(model, "models/model.pkl")

# input_schema = Schema(X_train)
# # Output schema ko simple float batana behtar hai
# output_schema = Schema(pd.DataFrame({'us_aqi': [10.5]})) 

# model_schema = ModelSchema(input_schema, output_schema)

# aqi_model = mr.python.create_model(
#     name=MODEL_NAME,
#     metrics={"mae": mae, "rmse": rmse, "r2": r2},
#     model_schema=model_schema,
#     input_example=X_train.sample(),
#     description="Voting Ensemble (XGB+LGBM+RF) with Version 2 Features"
# )

# aqi_model.save("models/model.pkl")

# print("🎉 Success! Ensemble Model is LIVE on Hopsworks.")

# # import hopsworks
# # import pandas as pd
# # import numpy as np
# # import xgboost as xgb
# # import lightgbm as lgb
# # from sklearn.ensemble import RandomForestRegressor, VotingRegressor
# # from sklearn.linear_model import LinearRegression, ElasticNet
# # from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
# # from hsml.schema import Schema
# # from hsml.model_schema import ModelSchema
# # import joblib
# # import os
# # from dotenv import load_dotenv

# # # --- LOAD SECRETS ---
# # load_dotenv()

# # # --- CONFIGURATION ---
# # HOPSWORKS_PROJECT = "aqi_predictor10"
# # FEATURE_VIEW_NAME = "aqi_view_main"
# # FEATURE_VIEW_VERSION = 2

# # print("🚀 Connecting to Hopsworks...")
# # try:
# #     project = hopsworks.login(project=HOPSWORKS_PROJECT)
# #     fs = project.get_feature_store()
# #     mr = project.get_model_registry()
# # except Exception as e:
# #     print(f"❌ Login Error: {e}")
# #     exit()

# # # 1. GET FEATURE VIEW
# # try:
# #     feature_view = fs.get_feature_view(name=FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)
# #     print("✅ Feature View Found.")
# # except:
# #     print("⚙️ Feature View not found. Please run backfill script.")
# #     exit()

# # # 2. CREATE TRAINING DATA (FIXED TIMEOUT ISSUE) 🛡️
# # print("📊 Creating Training Data...")
# # # Hum 'read_options' add kar rahe hain taake timeout na ho
# # X_train, X_test, y_train, y_test = feature_view.train_test_split(
# #     test_size=0.2, 
# #     description="Multi-Model Race Split",
# #     data_format="pandas",
# #     read_options={"use_hive": True}  # <--- YE CHANGE KIYA HAI (Slow but Safe)
# # )

# # X_train = X_train.drop(columns=["date"])
# # X_test = X_test.drop(columns=["date"])
# # y_train = y_train.values.ravel()
# # y_test = y_test.values.ravel()

# # print(f"✅ Data Ready! Train Shape: {X_train.shape}")

# # # 3. DEFINE MODELS 🤖
# # print("🤖 Defining Models...")

# # model_rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
# # model_xgb = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
# # model_lgb = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=10, random_state=42, verbose=-1)
# # model_lr = LinearRegression()
# # model_en = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)

# # model_voting = VotingRegressor(
# #     estimators=[('xgb', model_xgb), ('lgb', model_lgb), ('rf', model_rf)]
# # )

# # candidates = [
# #     ("RandomForest", model_rf),
# #     ("XGBoost", model_xgb),
# #     ("LightGBM", model_lgb),
# #     ("VotingEnsemble", model_voting),
# #     ("LinearRegression", model_lr),
# #     ("ElasticNet", model_en)
# # ]

# # # 4. TRAIN & RANK LOOP
# # print("\n🏎️ Starting Race...")
# # trained_models = []

# # for name, model in candidates:
# #     print(f"   Training {name}...")
# #     model.fit(X_train, y_train)
# #     preds = model.predict(X_test)
# #     rmse = np.sqrt(mean_squared_error(y_test, preds))
# #     mae = mean_absolute_error(y_test, preds)
# #     r2 = r2_score(y_test, preds)
    
# #     trained_models.append({
# #         "name": name, "model": model, "rmse": rmse, "mae": mae, "r2": r2
# #     })
# #     print(f"      -> RMSE: {rmse:.4f}")

# # # Sort & Pick Top 3
# # sorted_models = sorted(trained_models, key=lambda x: x['rmse'])
# # top_3_models = sorted_models[:3]

# # # 5. SAVE TO REGISTRY (FIXED UPLOAD ISSUE) 💾
# # print("\n💾 Saving Winners to Hopsworks...")
# # os.makedirs("models", exist_ok=True)

# # input_schema = Schema(X_train)
# # output_schema = Schema(pd.DataFrame({'us_aqi': [10.5]}))
# # model_schema = ModelSchema(input_schema, output_schema)

# # for rank, info in enumerate(top_3_models, start=1):
# #     model_name_reg = f"aqi_model_rank_{rank}"
# #     file_name = f"models/model_rank_{rank}.pkl"
    
# #     # A. Delete Old Model (Safai)
# #     try:
# #         old_model = mr.get_model(model_name_reg, version=None)
# #         old_model.delete()
# #         print(f"   🗑️ Replaced old {model_name_reg}")
# #     except:
# #         pass

# #     # B. Compress File (Ye zaroori hai taake upload fail na ho)
# #     joblib.dump(info['model'], file_name, compress=3) 
    
# #     # C. Upload
# #     print(f"   ☁️ Uploading Rank {rank}: {info['name']}...")
# #     hw_model = mr.python.create_model(
# #         name=model_name_reg,
# #         metrics={"rmse": info['rmse'], "mae": info['mae'], "r2": info['r2']},
# #         model_schema=model_schema,
# #         input_example=X_train.sample(),
# #         description=f"Rank {rank} Model: {info['name']}"
# #     )
# #     hw_model.save(file_name)
# #     print(f"   ✅ Saved Successfully!")

# # print("\n🎉 Training Complete!")

# # import hopsworks
# # import pandas as pd
# # import numpy as np
# # import xgboost as xgb
# # import lightgbm as lgb
# # from sklearn.ensemble import RandomForestRegressor, VotingRegressor
# # from sklearn.linear_model import LinearRegression
# # from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
# # from hsml.schema import Schema
# # from hsml.model_schema import ModelSchema
# # import joblib
# # import os
# # from dotenv import load_dotenv

# # # --- LOAD SECRETS ---
# # load_dotenv()

# # # --- CONFIGURATION ---
# # HOPSWORKS_PROJECT = "aqi_predictor10"
# # FEATURE_VIEW_NAME = "aqi_view_main"
# # FEATURE_VIEW_VERSION = 2
# # MODEL_NAME = "aqi_best_model" # Hum winner ko is naam se save karenge

# # print("🚀 Connecting to Hopsworks...")
# # try:
# #     project = hopsworks.login(project=HOPSWORKS_PROJECT)
# #     fs = project.get_feature_store()
# #     mr = project.get_model_registry()
# # except Exception as e:
# #     print(f"❌ Login Error: {e}")
# #     exit()

# # # 1. GET FEATURE VIEW
# # try:
# #     feature_view = fs.get_feature_view(name=FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)
# #     print("✅ Feature View Found.")
# # except:
# #     print("❌ Feature View not found. Backfill run karein.")
# #     exit()

# # # 2. CREATE TRAINING DATA (Standard Way)
# # print("📊 Creating Training Data...")
# # X_train, X_test, y_train, y_test = feature_view.train_test_split(
# #     test_size=0.2, 
# #     description="Model Comparison Split",
# #     data_format="pandas"
# # )

# # # Cleanup
# # X_train = X_train.drop(columns=["date"])
# # X_test = X_test.drop(columns=["date"])
# # y_train = y_train.values.ravel()
# # y_test = y_test.values.ravel()

# # print(f"✅ Data Ready! Train Shape: {X_train.shape}")

# # # 3. DEFINE CANDIDATES 🤖
# # print("🤖 Defining 5 Candidates...")

# # # Individual Models
# # rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
# # xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
# # lgbm = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=10, random_state=42, verbose=-1)
# # lr = LinearRegression()

# # # Ensemble (Sabka mix)
# # voting = VotingRegressor(
# #     estimators=[('xgb', xgb_model), ('lgb', lgbm), ('rf', rf)]
# # )

# # candidates = [
# #     ("RandomForest", rf),
# #     ("XGBoost", xgb_model),
# #     ("LightGBM", lgbm),
# #     ("LinearRegression", lr),
# #     ("VotingEnsemble", voting)
# # ]

# # # 4. TRAIN & EVALUATE LOOP
# # print("\n🏎️ Starting Race...")

# # best_model = None
# # best_rmse = float('inf') # Infinity (starting point)
# # best_metrics = {}
# # best_name = ""

# # for name, model in candidates:
# #     print(f"   🏃 Training {name}...")
    
# #     model.fit(X_train, y_train)
# #     preds = model.predict(X_test)
    
# #     # Calculate RMSE
# #     mse = mean_squared_error(y_test, preds)
# #     rmse = np.sqrt(mse)
# #     mae = mean_absolute_error(y_test, preds)
# #     r2 = r2_score(y_test, preds)
    
# #     print(f"      -> RMSE: {rmse:.4f}")
    
# #     # Check if this is the winner
# #     if rmse < best_rmse:
# #         best_rmse = rmse
# #         best_model = model
# #         best_name = name
# #         best_metrics = {"rmse": rmse, "mae": mae, "r2": r2}

# # print(f"\n🏆 WINNER: {best_name} (RMSE: {best_rmse:.4f})")

# # # 5. SAVE ONLY THE BEST MODEL (WITH REPLACEMENT) 💾
# # print(f"💾 Saving {best_name} to Hopsworks as '{MODEL_NAME}'...")

# # os.makedirs("models", exist_ok=True)
# # model_path = "models/best_model.pkl"

# # # --- REPLACEMENT LOGIC START ---
# # try:
# #     # Pehle check karo agar purana model hai to usay delete karo
# #     old_model = mr.get_model(MODEL_NAME, version=None)
# #     old_model.delete()
# #     print(f"   🗑️ Deleted old version of {MODEL_NAME}")
# # except:
# #     # Agar pehle se koi model nahi tha, to error ignore karo
# #     pass
# # # --- REPLACEMENT LOGIC END ---

# # # Save Local
# # joblib.dump(best_model, model_path)

# # # Schema
# # input_schema = Schema(X_train)
# # output_schema = Schema(pd.DataFrame({'us_aqi': [10.5]}))
# # model_schema = ModelSchema(input_schema, output_schema)

# # # Upload New
# # hw_model = mr.python.create_model(
# #     name=MODEL_NAME,
# #     metrics=best_metrics,
# #     model_schema=model_schema,
# #     input_example=X_train.sample(),
# #     description=f"Best Performing Model: {best_name}"
# # )

# # hw_model.save(model_path)
# # print("🎉 Best Model Replaced & Saved Successfully!")

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