from sklearn.cluster import KMeans, DBSCAN
import pandas as pd
from typing import Literal
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
def perform_clustering(df: pd.DataFrame, algorithm: Literal['kmeans', 'dbscan'], **kwargs):
    """
    Perform clustering on the provided DataFrame using the specified algorithm.

    Parameters:
    df (pd.DataFrame): The DataFrame to cluster.
    algorithm (str): The clustering algorithm to use ('kmeans' or 'dbscan').
    **kwargs: Additional keyword arguments to pass to the clustering algorithm.

    Returns:
    pd.DataFrame: DataFrame with an additional 'Cluster' column containing the cluster labels and the test score.
    dict: A dictionary containing the test score.
    """
    if algorithm == 'kmeans':
        model = KMeans(**kwargs)
        X = df.values
        labels = model.fit_predict(X)
        df['Cluster'] = labels
        test_score = model.score(X)
        return df, test_score
    elif algorithm == 'dbscan':
        eps = kwargs.get('eps', 0.5)
        min_samples = kwargs.get('min_samples', 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        X = df.values
        labels = model.fit_predict(X)
        df['Cluster'] = labels
        return df, None
    else:
        raise ValueError("Invalid algorithm. Please choose 'kmeans' or 'dbscan'.")

def perform_prediction(df: pd.DataFrame, target_column: str):

    models = [
    ('Regression Linear'),
    ('Regression Ridge'),
    ('Regression Lasso'),
    ('K-Nearest Neighbors'),
    ('Decision Tree'),
    ]

    resultats = []
    for name in models:
        
        mse, r2, y_pred = predict(name, df, target_column)
        st.write(f"{name} :MSE {mse}, R2 {r2}")
        resultats.append((name, mse, r2))
    resultats = pd.DataFrame(resultats, columns=['Model', 'MSE', 'R2'])
    resultats = resultats.sort_values(by='MSE', ascending=True)
    
    st.write(resultats)
    return resultats


    

def predict(model_name, df: pd.DataFrame, target_column: str):
    models = [
    ('Regression Linear', LinearRegression()),
    ('Regression Ridge', Ridge()),
    ('Regression Lasso', Lasso()),
    ('K-Nearest Neighbors', KNeighborsRegressor()),
    ('Decision Tree', DecisionTreeRegressor()),
    ]
    model = next((m for m_name, m in models if m_name == model_name), None)
    
    x=df.drop(target_column, axis=1).values
    y=df[target_column].values
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)  # entraine le modele
    y_pred = model.predict(X_test)  # utilise le modele entraine pour predire les valeurs

    mse = mean_squared_error(y_test, y_pred)  # calcul l'erreur quadratique avec la bibliotheque scikit-learn
    r2 = r2_score(y_test, y_pred)  # calcul le coeff de determination R2 avec la bibliotheque scikit-learn

    return mse, r2, y_pred  # retourne les valeurs de l'erreur quadratique et du coeff de determination
