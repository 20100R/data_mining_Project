from sklearn.cluster import KMeans, DBSCAN
import pandas as pd
from typing import Literal
def perform_clustering(df: pd.DataFrame, algorithm: Literal['kmeans', 'dbscan'], **kwargs):
    """
    Perform clustering on the provided DataFrame using the specified algorithm.

    Parameters:
    df (pd.DataFrame): The DataFrame to cluster.
    algorithm (str): The clustering algorithm to use ('kmeans' or 'dbscan').
    **kwargs: Additional keyword arguments to pass to the clustering algorithm.

    Returns:
    pd.Series: Cluster labels for each data point.
    """
    if algorithm == 'kmeans':
        model = KMeans(**kwargs)
    elif algorithm == 'dbscan':
        model = DBSCAN(**kwargs)
    else:
        raise ValueError("Invalid algorithm. Please choose 'kmeans' or 'dbscan'.")
    
    cluster_labels = model.fit_predict(df)
    return pd.Series(cluster_labels, index=df.index)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd

def perform_prediction(df: pd.DataFrame, target_column: str, algorithm: Literal['linear_regression', 'decision_tree'], **kwargs):
    """
    Perform prediction on the provided DataFrame using the specified algorithm.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the features and the target column.
    target_column (str): The name of the target column in the DataFrame.
    algorithm (str): The prediction algorithm to use ('linear_regression' or 'decision_tree').
    **kwargs: Additional keyword arguments to pass to the prediction algorithm.

    Returns:
    model: Trained model object.
    dict: Dictionary containing 'train_score' and 'test_score'.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if algorithm == 'linear_regression':
        model = LinearRegression(**kwargs)
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        return model, {'train_score': train_score, 'test_score': test_score, 'metric': 'R^2 Score'}
    
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