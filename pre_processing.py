import pandas as pd
from typing import Literal
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale

from sklearn.impute import KNNImputer

def delete_missing_row_col(df: pd.DataFrame) -> pd.DataFrame:
    # Supprimer les colonnes contenant au moins une valeur NaN
    df = df.dropna(axis=1, how='any')
    # Supprimer les lignes contenant au moins une valeur NaN
    df = df.dropna(axis=0, how='any')
    return df

def replace_missing_values(df: pd.DataFrame, method: Literal['mean', 'median', 'mode', 'knn']) -> pd.DataFrame:
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
    
def normalize_min_max(df:pd.DataFrame):
    """
    Normalize a DataFrame using Min-Max normalization.

    Parameters:
    df (pd.DataFrame): The DataFrame to normalize.

    Returns:
    pd.DataFrame: The normalized DataFrame.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

def normalize_z_standardization(df:pd.DataFrame):
    """
    Normalize a DataFrame using Z-standardization. 
    Which resizes the data to have a mean of 0 and a standard
    deviation of 1.

    Parameters:
    df (pd.DataFrame): The DataFrame to normalize.

    Returns:
    pd.DataFrame: The normalized DataFrame.
    """
    df_normalized = scale (df)
    return pd.DataFrame(df_normalized, columns=df.columns)

