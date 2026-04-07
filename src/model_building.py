import pandas as pd
import numpy as np


def prepare_data_for_modeling(df):
  df = df.sort_values("Date")
  X = df.drop(["Boardings_log","Date"],axis=1)
  y = df["Boardings_log"]
  train = df[df["Year"] < 2024]
  test = df[df["Year"] >= 2024]
  X_train = train[X.columns]
  y_train = train[y.name]
  X_test = test[X.columns]
  y_test = test[y.name]
  return X_train,y_train,X_test,y_test

def baseline_model(X_test):
   # Naive baseline: predict using previous hour's boardings
   y_pred = np.log1p(X_test["lag_1_row"]).fillna(0)
   return y_pred

def gradient_boosting_model(X_train,y_train,X_test,a,b, c):
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(n_estimators=a,learning_rate=b,max_depth=c,random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def xgboost_model(X_train,y_train,X_test,a, b, c,):
    from  xgboost  import XGBRegressor
    model = XGBRegressor(n_estimators=a,learning_rate=b,max_depth=c, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model,y_pred

def evaluate_model(y_test,y_pred):
    y_pred_actual = np.expm1(y_pred)
    y_test_actual = np.expm1(y_test)
    from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    mse = mean_squared_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_actual, y_pred_actual)
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")



def feature_importance(model, X_train, threshold=0.01):
 import plotly.express as px
 import plotly.io as pio

 # Ensure GitHub rendering
 pio.renderers.default = "png"

 importance = model.feature_importances_
 feature_names = X_train.columns

 importance_df = pd.DataFrame({
     "Feature": feature_names,
     "Importance": importance
    })

 # 🔥 Filter features above threshold (1%)
 importance_df = importance_df[importance_df["Importance"] >= threshold]

 # Sort after filtering
 importance_df = importance_df.sort_values("Importance", ascending=True)

 fig = px.bar(
     importance_df,
     x="Importance",
     y="Feature",
     orientation="h",
     text_auto=".2%",
     title="Feature Importance (>1%)"
    )

 return  fig.show("png")  # force static for GitHub