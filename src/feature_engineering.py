import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def sorting_datetime(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def aggregate_daily_boardings(df):
    return df.groupby("Date", as_index=False)["Boardings"].sum().sort_values("Date").reset_index(drop=True)

def create_lag_1_row(df):
    df["lag_1_row"] = df.groupby(["Route", "Direction", "Stop"])["Boardings"].shift(1)
    return df

# -----------------------------
# Lag Features(Core)
# -----------------------------
def create_lag_features(df,lagdays):
    for lag in lagdays:
        df[f"Boardings_lag_{lag}"] = df["Boardings"].shift(lag)
        df[f"Boardings_lag_{lag}"] = df[f"Boardings_lag_{lag}"].astype("Int32")
    return df

# -----------------------------
# Rolling Statistics
# (shifted to avoid leakage)
# -----------------------------
def create_rolling_features(df,window_sizes,x):
    for window in window_sizes:
        df[f"Boardings_roll_mean_{window}"] = df["Boardings"].shift(1).rolling(window=window).mean()
    df[f"Boardings_roll_std_{x}"] = df["Boardings"].shift(1).rolling(window=x).std()
    return df

# -----------------------------
# Calendar Features
# -----------------------------
def create_datetime_features(df):
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["is_weekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
    return df

def merge_dataframes(df1,df2,on="Date"):
    return df1.merge(df2,on=on,how="left",suffixes=("","_daily"))

def feature_engineering(df,lagdays,window_sizes,x):
    df = sorting_datetime(df)
    df = create_datetime_features(df)
    df = create_lag_1_row(df)
    daily_df = aggregate_daily_boardings(df).shift(1)
    daily_df = create_datetime_features(daily_df)
    daily_df = create_lag_features(daily_df, lagdays)
    daily_df = create_rolling_features(daily_df, window_sizes, x)
    # -----------------------------
    # Drop rows with NaNs from lagging
    # -----------------------------
    daily_df = daily_df.dropna().reset_index(drop=True)
    df=merge_dataframes(df,daily_df)
    df = df.drop(columns=["DayOfWeek_daily","Month_daily","Year_daily","is_weekend_daily",'TypeDay'])
    df = df.dropna().reset_index(drop=True)
    # -----------------------------
    # Cyclical Encoding for Hour of Day 
    # -----------------------------
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    # -----------------------------
    # Encoding Categorical Variables
    # -----------------------------
    df = pd.get_dummies(df,columns=["Direction"],drop_first=True)
    le_route = LabelEncoder()
    le_stop = LabelEncoder()
    df["Route"] = le_route.fit_transform(df["Route"])
    df["Stop"] = le_stop.fit_transform(df["Stop"])
    # -----------------------------
    # Log Transformation of Target Variable
    # -----------------------------
    df["Boardings_log"] = np.log1p(df["Boardings"])
    df=df.drop(columns='Boardings')
    return df


