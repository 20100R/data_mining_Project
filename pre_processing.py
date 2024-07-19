import pandas as pd
from typing import Literal
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

from sklearn.impute import KNNImputer

def delete_missing_row_col(df: pd.DataFrame):
    df = df.dropna(axis=1, how='any')
    df = df.dropna(axis=0, how='any')
    return df

def replace_missing_values(df: pd.DataFrame, method: Literal['mean', 'median', 'mode', 'knn']):
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns

    if method == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif method == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif method == 'mode':
        # Calculate mode for each numeric column and use the first mode if there are multiple modes
        for col in numeric_cols:
            mode_value = df[col].mode().iloc[0]  # This takes the first mode if there are multiple
            df[col] = df[col].fillna(mode_value)
    elif method == 'knn':
        # Use KNN imputer (default 5)
        imputer = KNNImputer(n_neighbors=5)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    else:
        raise ValueError("Invalid method. Please choose 'mean', 'median', 'mode', or 'knn'.")

    return df
    
def normalize_min_max(df: pd.DataFrame):
    # Create a scaler object
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Identify numeric columns 
    numeric_cols = df.select_dtypes(include=['number']).columns

    df_normalized = df.copy()
    df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return pd.DataFrame(df_normalized, columns=df.columns)

def normalize_z_standardization(df: pd.DataFrame):
    # Create a scaler object for Z-score standardization
    scaler = StandardScaler()
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    df_normalized = df.copy()
    df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return pd.DataFrame(df_normalized, columns=df.columns)

def normalize_robust(df: pd.DataFrame):
    """
    Robust Scaling removes the median and scales the data according to the Interquartile Range (IQR).
    This method is less sensitive to outliers than other scaling methods.

    """
    # Create a robust scaler object
    scaler = RobustScaler()
    
    # Identify numeric columns (select types that are considered numeric)
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    df_normalized = df.copy()
    df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return pd.DataFrame(df_normalized, columns=df.columns)

