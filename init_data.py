import pandas as pd
import numpy as np

def load_csv(file_path,header=0,sep=','):

    return pd.read_csv(file_path,header=header,sep=sep)
def display_description(data:pd.DataFrame):
    #display the first and the last row of the data
    print(data.head())
    print(data.tail())
def statistical_summary(data:pd.DataFrame): 
    """: Provide a basic statistical summary of the data, including the
    number of lines and columns, the name of the columns, the number of missing values per
    column, etc."""
    print(data.describe())
    print(data.info())
    print(data.isnull().sum())

