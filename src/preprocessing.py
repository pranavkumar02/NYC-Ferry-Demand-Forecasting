import pandas as pd
import numpy as np

def loading_data(file_path):
    # Load the dataset
 df = pd.read_csv(file_path,
    parse_dates=["Date"],
    thousands=",",
    dtype={
        "Hour": "Int8",
        "Route": "category",
        "Direction": "category",
        "Stop": "category",
        "TypeDay": "category"
    }
    )
 return df

def clean_data(df):
 # Remove commas and convert Boardings to numeric
 df["Boardings"] = (df["Boardings"].astype(str).str.replace(",", "", regex=False))
 df["Boardings"] = pd.to_numeric(df["Boardings"],errors="coerce").astype("Int32")
 # Replace "(blank)" with NaN in "Direction" column
 df["Direction"] = df["Direction"].replace("(blank)", pd.NA)
 # Remove rows with missing values 
 df = df.dropna()
 # Remove duplicates
 df = df.drop_duplicates()
 return df

def preprocess_data(file_path):
    df = loading_data(file_path)
    df = clean_data(df)
    return df

def understand_data(df):
    print("\nDataFrame Shape:", df.shape)
    print("\n\nDataFrame Info:")
    print(df.info())
    print("\n\nDataFrame Description:")
    print(df.describe(include="all"))
    print("\n\nUnique Values in Each Column:")
    for column in df.columns:
        unique_values = df[column].unique()
        print(f"Unique values in column '{column}': {unique_values}\n")
    print('\n\nFirst 10 rows of the DataFrame:')
    print(df.head(10))

def saving_cleaned_data(df,OUTPUT_PATH):
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Cleaned dataset saved to {OUTPUT_PATH}")

