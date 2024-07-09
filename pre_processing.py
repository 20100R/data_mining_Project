import pandas as pd
from typing import Literal
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