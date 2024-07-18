import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def histograms(data: pd.DataFrame):
    for col in data.columns:
        st.subheader(f"Histogram for {col}")
        fig, ax = plt.subplots()
        
        if data[col].dtype == 'bool' or data[col].dtype == 'object':
            # For categorical data or booleans, count the frequency of each
            data[col].value_counts().plot(kind='bar', ax=ax)
        else:
            # Compute mean and standard deviation
            mean = data[col].mean()
            std_dev = data[col].std()
            
            # Filter data within 3 standard deviations from the mean
            filtered_data = data[col][(data[col] >= mean - 3 * std_dev) & (data[col] <= mean + 3 * std_dev)]
            filtered_data.hist(ax=ax)

        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

def box_plots(data: pd.DataFrame):
    # Check if there are any numeric columns in the DataFrame
    numeric_cols = data.select_dtypes(include=['number']).columns
    if numeric_cols.empty:
        st.write("No numeric data to plot.")
    else:
        for col in numeric_cols:
            st.subheader(f"Box Plot for {col}")
            fig, ax = plt.subplots()

            # Compute mean and standard deviation
            mean = data[col].mean()
            std_dev = data[col].std()
            
            # Filter data within 3 standard deviations from the mean
            filtered_data = data[col][(data[col] >= mean - 3 * std_dev) & (data[col] <= mean + 3 * std_dev)]
            
            filtered_data.plot(kind='box', ax=ax)
            ax.set_ylabel('Values')
            st.pyplot(fig)


def correlation_heatmap(data: pd.DataFrame):
    # Filter out non-numeric columns to ensure the correlation matrix is calculated correctly
    numeric_data = data.select_dtypes(include=[np.number])

    # Calculate the correlation matrix for numeric columns
    corr_matrix = numeric_data.corr()

    # Plotting the heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))  
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix Heatmap')
    st.pyplot(fig)


def plot_clusters(data:pd.DataFrame,target_column:str,target_column2:str):
    """
    Display a scatter plot of the clustered data.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the cluster labels.
    """
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(data[target_column], data[target_column2], c=data["Cluster"], cmap='viridis')
    
    # Add a color bar
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    
    ax.set_xlabel(target_column)
    ax.set_ylabel(target_column2)
    ax.set_title("Clustered Data")
    st.pyplot(fig)