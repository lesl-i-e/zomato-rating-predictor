# 🍽️ Zomato Restaurant Rating Predictor

An end-to-end machine learning project that predicts a restaurant's Zomato rating based on its attributes — and explains *why* using SHAP interpretability.

Built with XGBoost, trained on 7,400+ restaurants across 15 countries, deployed as an interactive Streamlit dashboard.

---

## Project Structure

```
zomato-rating-predictor/
│
├── data/
│   ├── zomato.csv                        # Raw dataset (Kaggle)
│   └── zomato_clean.csv                  # Cleaned, feature-engineered dataset (output of Notebook 1)
│
├── notebooks/
│   ├── 01_cleaning_and_features.ipynb    # Data cleaning & feature engineering
│   ├── 02_eda.ipynb                      # Exploratory data analysis (9 visualisations)
│   └── 03_modelling_and_shap.ipynb       # Model training, comparison & SHAP interpretability
│
├── models/
│   ├── xgb_model.pkl                     # Trained XGBoost model
│   ├── feature_columns.pkl               # Feature column order (required by Streamlit app)
│   └── best_params.pkl                   # Best hyperparameters from RandomizedSearchCV
│
├── app/
│   └── streamlit_app.py                  # Streamlit dashboard
│
├── .gitignore
└── README.md
```

---

## Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression (baseline) | 0.3698 | 0.2731 | 0.5578 |
| Random Forest | 0.3395 | 0.2529 | 0.6274 |
| **XGBoost (tuned)** | **0.3317** | **0.2479** | **0.6442** |

**Top features by SHAP importance:**
1. Votes (review count)
2. City frequency (how restaurant-dense the city is)
3. Cuisine frequency (how common the cuisine is on Zomato)
4. Log Votes
5. Average Cost for Two

---

## Notebooks

### 01 — Cleaning & Feature Engineering
- Drops leaky and irrelevant columns
- Removes 2,148 unrated restaurants (`Aggregate rating == 0`)
- Engineers 6 new features: `cuisine_count`, `is_multi_cuisine`, `log_votes`, `cost_per_person`, `has_both_services`, `country_name`
- Encodes categoricals: one-hot (country), frequency encoding (city, cuisine)

### 02 — EDA
9 visualisations covering:
- Target distribution (rating histogram + boxplot)
- Rating by country, price range, service features
- Votes vs rating (raw and log-transformed)
- Top 20 cities by restaurant count
- Top cuisines by average rating
- Correlation heatmap

### 03 — Modelling & SHAP
- Linear Regression baseline → Random Forest → XGBoost with 60-iteration RandomizedSearchCV
- Residual analysis (predicted vs actual, residual distribution)
- SHAP beeswarm (global importance), bar chart, waterfall (individual explanations), dependence plot

---

## Streamlit App

**Features:**
- Sidebar inputs: city, cuisine, country, price range, cost, services, cuisine variety, votes
- Predicted rating with colour-coded tier (Excellent / Very Good / Good / Average / Poor)
- SHAP bar chart for the specific prediction — which features pushed it up or down
- Plain-English business insights for each key driver
- Dataset explorer: rating distribution, top cities, top cuisines

**Run locally:**
```bash
pip install streamlit xgboost shap pandas numpy matplotlib
streamlit run app/streamlit_app.py
```

**Run on Google Colab:**
```python
!pip install streamlit pyngrok shap xgboost --quiet
from pyngrok import ngrok
!streamlit run "/content/drive/MyDrive/Colab Notebooks/zomato-rating-predictor/app/streamlit_app.py" &
public_url = ngrok.connect(8501)
print(f"Live at: {public_url}")
```

---

## Dataset

**Source:** [Zomato Restaurants Dataset — Kaggle](https://www.kaggle.com/datasets/shrutimehta/zomato-restaurants-data)

- 9,551 restaurants across 15 countries
- 7,403 rated restaurants used after cleaning
- Features: location, cuisine, price range, services, votes, cost

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-E84545?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-Interpretability-0F6E56?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)

---

## Author

**Gedion Leslie Kweya Odera**  
Data Scientist → Data Engineer · Nairobi, Kenya  
[LinkedIn](https://www.linkedin.com/in/gideon-leslie-385949253/) · [GitHub](https://github.com/lesl-i-e) · gideonleslie9@gmail.com
