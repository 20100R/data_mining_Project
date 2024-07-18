import streamlit as st
import pandas as pd
from typing import Literal
import matplotlib.pyplot as plt

def histograms(data: pd.DataFrame):
    for col in data.columns:
        st.subheader(f"Histogram for {col}")
        fig, ax = plt.subplots()
        data[col].hist(ax=ax)
        ax.set_xlabel(f'{col}') 
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

def box_plots(data: pd.DataFrame):
    for col in data.columns:
        st.subheader(f"Box Plot for {col}")
        fig, ax = plt.subplots()
        data[col].plot(kind='box', ax=ax)
        st.pyplot(fig)
        
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