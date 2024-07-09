import pandas as pd
from typing import Literal
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale

def delete_empty_row_col(df:pd.DataFrame):
    """
    Remove completely empty rows and columns from a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.

    Returns:
    pd.DataFrame: The DataFrame without completely empty rows and columns.
    """
    # Supprimer les colonnes entièrement vides
    df = df.dropna(axis=1, how='all')
    # Supprimer les lignes entièrement vides
    df = df.dropna(axis=0, how='all')
    return df

def replace_missing_values(df, method=Literal['mean', 'median', 'mode']):
    """
    Replace missing values in a DataFrame with a specified value.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    method (str): The method to use to replace missing values. Options are 'mean', 'median', 'mode'.

    Returns:
    pd.DataFrame: The DataFrame with missing values replaced.
    """
    if method == 'mean':
        return df.fillna(df.mean())
    elif method == 'median':
        return df.fillna(df.median())
    elif method == 'mode': #TODO # je suis pas sur de ca à vérifier
        return df.fillna(df.mode())
    else:
        raise ValueError("Invalid method. Please choose 'mean', 'median', or 'mode'.")
    
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

