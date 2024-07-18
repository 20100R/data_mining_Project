from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
import pandas as pd
from typing import Literal
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

def cluster_spectral(data, n_clusters=3):
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
    labels = spectral.fit_predict(data)
    return labels

def cluster_kmeans(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(data)
    return labels, kmeans.cluster_centers_

def cluster_dbscan(data, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    return labels

def perform_clustering(data, algorithm, **kwargs):
    if algorithm == 'kmeans':
        return cluster_kmeans(data, **kwargs)
    elif algorithm == 'dbscan':
        return cluster_dbscan(data, **kwargs)
    elif algorithm == 'spectral':
        return cluster_spectral(data, **kwargs)
    else:
        raise ValueError("Unsupported clustering algorithm")
    
def compute_cluster_stats(data, labels, centers=None):
    """
    Compute and return statistics for clusters.
    """
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    
    # Determine the number of clusters
    unique_labels = np.unique(labels[labels >= 0])  # Excluding noise if present
    stats = []

    for label in unique_labels:
        cluster_data = data[labels == label]
        cluster_size = cluster_data.shape[0]
        if centers is not None:
            center = centers[label]
            # Calculate mean distance to the center
            distances = np.sqrt(np.sum((cluster_data - center)**2, axis=1))
            mean_distance = np.mean(distances)
            stats.append({'Cluster': label, 'Size': cluster_size, 'Center': center, 'Mean Distance': mean_distance})
        else:
            stats.append({'Cluster': label, 'Size': cluster_size})
    
    # Handle noise specially for DBSCAN
    if -1 in labels:
        noise_size = (labels == -1).sum()
        stats.append({'Cluster': 'Noise', 'Size': noise_size})

    return pd.DataFrame(stats)


def perform_prediction(df: pd.DataFrame, target_column: str, algorithm: Literal['linear_regression', 'decision_tree'], **kwargs):
    """
    Perform prediction on the provided DataFrame using the specified algorithm.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the features and the target column.
    target_column (str): The name of the target column in the DataFrame.
    algorithm (str): The prediction algorithm to use ('linear_regression' or 'decision_tree').
    **kwargs: Additional keyword arguments to pass to the prediction algorithm.

    Returns:
    model: The trained model.
    dict: A dictionary containing the training and test scores and the metric used.
    """
    y = df[target_column]
    X = df.drop(columns=[target_column])
 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if algorithm == 'linear_regression':
        model = LinearRegression(**kwargs)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_score = mean_squared_error(y_train, y_train_pred)
        test_score = mean_squared_error(y_test, y_test_pred)
        return model, {'train_score': train_score, 'test_score': test_score, 'metric': 'Mean Squared Error'}    
    elif algorithm == 'decision_tree':
        model = DecisionTreeClassifier(**kwargs)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_score = accuracy_score(y_train, y_train_pred)
        test_score = accuracy_score(y_test, y_test_pred)
        return model, {'train_score': train_score, 'test_score': test_score, 'metric': 'Accuracy'}
    
    else:
        raise ValueError("Invalid algorithm. Please choose 'linear_regression' or 'decision_tree'.")
