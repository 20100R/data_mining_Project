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

    st.write(data["Cluster"])
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(data[target_column], data[target_column2], c=data["Cluster"], cmap='viridis')
    
    # Add a color bar
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    
    ax.set_xlabel(target_column)
    ax.set_ylabel(target_column2)
    ax.set_title("Clustered Data")
    st.pyplot(fig)