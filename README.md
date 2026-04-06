# NYC Ferry Ridership Forecasting

Forecasting daily NYC Ferry ridership using historical transit data and machine learning.

---

## 🚀 Project Overview
This project builds an end-to-end time-series forecasting pipeline to model **daily NYC Ferry ridership**.  
The goal is to understand demand patterns and establish a defensible baseline for forecasting using machine learning.

The workflow follows a production-style structure:
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Baseline and ensemble modeling
- Evaluation against a naïve benchmark

---

## 📊 Data
The raw dataset contains granular NYC Ferry ridership records with route-, stop-, direction-, and time-level detail.
For modeling:
- Data is **aggregated to daily total ridership**
- This produces a stable system-level time series suitable for forecasting

---

## 🔍 Exploratory Data Analysis
Key EDA findings:
- Strong temporal dependence in ridership
- Clear weekly and yearly seasonality
- Persistence effects where recent ridership strongly influences future demand
- Temporary disruptions followed by recovery

These insights motivated the use of lag-based and rolling statistical features.

---

## 🛠 Feature Engineering
Features were engineered on the daily aggregated dataset to capture temporal structure:

- **Calendar features**
  - Year, month, day of week, weekend indicator
- **Lag features**
  - 1-day, 7-day, 14-day, and 30-day lags
- **Rolling statistics**
  - 7-day and 30-day rolling means
  - 7-day rolling standard deviation

All lag and rolling features are shifted to prevent target leakage.

Final dataset:
- ~3,000 daily observations
- 12 features + target (`Boardings`)

---

## 🤖 Modeling
Models evaluated:

1. **Naïve Persistence Baseline**
   - Predicts today’s ridership as yesterday’s value
2. **Random Forest Regressor**
   - Nonlinear ensemble baseline
3. **Gradient Boosting Regressor**
   - Strong tabular data benchmark

---

## 📈 Results
Models were evaluated using **Mean Absolute Error (MAE)**.

| Model | MAE |
|------|-----|
| Naïve Persistence (Lag-1) | ~4,371 |
| Random Forest | ~3,232 |
| Gradient Boosting | ~3,272 |

Ensemble models improve MAE by **~25%** over the naïve baseline, indicating meaningful signal captured by engineered features.

---

## 🧠 Key Takeaways
- Aggregation is critical when working with highly granular transit data
- Lag and rolling features are essential for time-series forecasting with ML
- Feature quality has a larger impact than model complexity
- Ensemble models significantly outperform naïve persistence

---

## ⚠️ Limitations & Future Work
- Use time-aware train/test splits
- Add external data (weather, holidays, service changes)
- Explore route-level or hierarchical forecasting
- Compare against specialized time-series models

---

## 📁 Project Structure

Python, Pandas, Scikit-learn, XGBoost, Tableau, VS Code
├── notebooks/
│ ├── 01_eda_nyc_ferry.ipynb
│ ├── 02_feature_engineering.ipynb
│ └── 03_model_training.ipynb
└── README.md

---

## 📦 Data Availability

The dataset is excluded from this repository due to size constraints.
It can be obtained from https://data.cityofnewyork.us/Transportation/NYC-Ferry-Ridership/t5n6-gx8c/about_data or provided upon request.

---

## ▶️ How to Run
```bash
pip install pandas numpy scikit-learn

