# 🌆 Karachi AQI Predictor (End-to-End MLOps)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Hopsworks](https://img.shields.io/badge/Feature%20Store-Hopsworks-green)
![GitHub Actions](https://img.shields.io/badge/Automation-GitHub%20Actions-orange)

An End-to-End Machine Learning project that predicts the **Air Quality Index (AQI)** for Karachi. This system is fully automated: it fetches data hourly, retrains models daily to find the best performer, and serves predictions via a Streamlit dashboard.

🔗 **Live Demo:** https://aqiproject.streamlit.app/

---

## 🧠 Key Features (Smart MLOps)

* **🔄 Automated Data Pipeline:** Runs hourly to fetch real-time weather & pollution data from OpenMeteo API.
* **🏆 Champion Model Selection:** The training pipeline trains multiple algorithms (e.g., XGBoost, Random Forest, Linear Regression), compares their performance, and automatically registers only the **Best Model** to the registry.
* **🗄️ Feature Store:** Uses **Hopsworks** to manage historical data (Backfill) and real-time features.
* **🤖 CI/CD Automation:** **GitHub Actions** handles the scheduling of data ingestion and model retraining without human intervention.
* **📊 Interactive Dashboard:** A user-friendly interface built with **Streamlit** to visualize current AQI and 3-day forecasts.

---

## 📂 Project Structure & File Description

Here is the breakdown of the files in this repository:

| File / Folder | Description |
| :--- | :--- |
| **`.github/workflows/`** | Contains YAML files for automation (`hourly_data.yml` & `daily_training_prediction.yml`). |
| **`models/`** | Stores the locally saved model artifacts (e.g., `best_model.pkl`). |
| **`1_fetch_data.py`** | Script to test API connections and fetch raw data from OpenMeteo. |
| **`2_backfill_feature_store.py`** | **Historical Data:** Used one-time to upload historical weather/AQI data to Hopsworks for initial training. |
| **`3_feature_pipeline.py`** | **Hourly Ingestion:** The operational script that fetches *new* data every hour and pushes it to the Feature Store. |
| **`4_model_training.py`** | **Training Engine:** Trains multiple models, evaluates them, selects the **Champion Model**, and saves it to the Hopsworks Model Registry. |
| **`5_batch_prediction.py`** | **Inference:** Loads the best model, generates forecasts for the next 3 days, and saves results for the App. |
| **`app.py`** | **Frontend:** The Streamlit application source code. |
| **`eda.ipynb`** | **Analysis:** Jupyter Notebook for Exploratory Data Analysis (EDA) and data visualization. |
| **`aqi_data_for_eda.csv`** | Sample CSV data used for local analysis and testing. |
| **`requirements.txt`** | List of all Python libraries required to run the project. |

---

## 🏗️ Automation Pipelines (GitHub Actions)

This project uses **GitHub Actions** for serverless automation:

| Workflow File | Schedule | Function |
| :--- | :--- | :--- |
| `hourly_data.yml` | **Every Hour** | Runs `3_feature_pipeline.py` to keep the Feature Store updated with the latest readings. |
| `daily_training_prediction.yml` | **Daily (UTC 00:00)** | Runs `4_model_training.py` (Retrain & Select Best) followed by `5_batch_prediction.py` (Update Forecast). |

---

## 🚀 How to Run Locally

If you want to run this project on your own machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Muhammad867/AQI_Project.git](https://github.com/Muhammad867/AQI_Project.git)
    cd AQI_Project
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables:**
    Create a `.env` file in the root directory and add your Hopsworks API Key:
    ```
    HOPSWORKS_API_KEY=your_api_key_here
    ```

4.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

---

## 🛠️ Tech Stack

* **Language:** Python 3.10
* **Data Processing:** Pandas, Numpy
* **Machine Learning:** Scikit-Learn, XGBoost, LightGBM
* **MLOps & Feature Store:** Hopsworks, Confluent-Kafka, Joblib
* **API Integration:** OpenMeteo-Requests, Retry-Requests
* **Frontend:** Streamlit
* **Utilities:** Python-Dotenv
* **Automation:** GitHub Actions

---
*Created by Muhammad*
