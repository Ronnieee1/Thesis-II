import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def extract_lexical_features(df):
    # Example: Extract lexical features from URLs
    # ...implement feature extraction logic...
    return df

def clean_and_scale(df):
    # Remove duplicates, missing values, outliers
    df = df.drop_duplicates().dropna()
    # ...outlier removal logic...
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return pd.DataFrame(df_scaled, columns=df.columns)

def apply_smote(X, y):
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res
