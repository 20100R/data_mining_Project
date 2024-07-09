import streamlit as st
import pandas as pd
from typing import Literal

def histograms(data:pd.DataFrame):
    """
    Display histograms for each column in a DataFrame.

    Parameters:
    data (pd.DataFrame): The DataFrame to display histograms for.
    """
    for col in data.columns:
        st.write(f"## {col}")
        st.write(data[col].hist())
        st.pyplot()

def box_plots(data:pd.DataFrame):
    """
    Display box plots for each column in a DataFrame.

    Parameters:
    data (pd.DataFrame): The DataFrame to display box plots for.
    """
    for col in data.columns:
        st.write(f"## {col}")
        st.write(data[col].plot(kind='box'))
        st.pyplot()
        