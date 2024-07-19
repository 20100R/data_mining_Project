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


def plot_clusters(data, labels, feature_names, centers=None):
    
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()  # Convert DataFrame to NumPy array if necessary

    fig, ax = plt.subplots()
    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    ax.set_title('Cluster Results')
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])    
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], s=200, c='red', marker='X')  # mark cluster centers
    
    st.pyplot(fig)

def plot_prediction(resultats:pd.DataFrame):
    fig, z = plt.subplots(figsize=(16, 12))
    barre = 0.3
    x = np.arange(len(resultats))

    z.bar(x + barre, resultats['MSE'], width=barre, label='MSE')
    z.bar(x - barre, resultats['R2'], width=barre, label='R2')

    z.legend()

    z.set_xticks(x)
    z.set_xticklabels(resultats['Model'], rotation=45)
    z.legend()

    plt.title("comparaison d'efficacite des modeles")
    st.pyplot(fig)

def plot_predict(y_test, y_pred):
                #afficher les predictions
            fig = plt.figure()
            plt.plot(y_test, y_pred, 'o')
            plt.xlabel('True values')
            plt.ylabel('Predicted values')
            plt.title('True vs Predicted values')
            st.pyplot(fig)

def plot_feature_importance(feature_importance, features):
    fig, ax = plt.subplots()
    ax.bar(features, feature_importance)
    ax.set_xlabel('Feature')
    ax.set_ylabel('Importance')
    ax.set_title('Feature Importance')
    ax.set_xticklabels(features, rotation=45)
    st.pyplot(fig)