import streamlit as st
import pandas as pd
from typing import Literal
import matplotlib.pyplot as plt
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
        
def plot_clusters(data:pd.DataFrame,target_column:str,target_column2:str):
    """
    Display a scatter plot of the clustered data.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the cluster labels.
    """
    fig = plt.figure()
    plt.scatter(data.iloc[:,0], data.iloc[:,1], c=data['Cluster'])
    plt.xlabel(target_column)
    plt.ylabel(target_column2)
    plt.title("Clustered Data")
    st.pyplot(fig)