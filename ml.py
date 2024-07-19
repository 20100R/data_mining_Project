from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
import streamlit as st
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

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
        
        mse, r2, y_pred, feature_importance = predict(name, df, target_column)
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
    colums = df.columns
    colums = colums.drop(target_column)

    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)  # entraine le modele
    y_pred = model.predict(X_test)  # utilise le modele entraine pour predire les valeurs

    mse = mean_squared_error(y_test, y_pred)  # calcul l'erreur quadratique avec la bibliotheque scikit-learn
    r2 = r2_score(y_test, y_pred)  # calcul le coeff de determination R2 avec la bibliotheque scikit-learn
    if model_name in ['Regression Linear', 'Regression Ridge', 'Regression Lasso']:
        importances = model.coef_
        feature_importance = pd.Series(importances, index=colums)
    
    elif model_name == 'Decision Tree':
        importances = model.feature_importances_
        feature_importance = pd.Series(importances, index=colums)
        fig, ax = plt.subplots(figsize=(15, 10))
        plot_tree(model, feature_names=df.drop(columns=[target_column]).columns, filled=True, ax=ax,max_depth=6)
        plt.title(f"Decision Tree for {target_column}")
        st.pyplot(fig)
    elif model_name == 'K-Nearest Neighbors':
        from sklearn.decomposition import PCA
        result = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42)
        feature_importance = pd.Series(result.importances_mean, index=colums)
        pca = PCA(n_components=2)
        X_test_pca = pca.fit_transform(X_test)
        fig = plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred, cmap='viridis')
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('')
        plt.ylabel('')
        st.pyplot(fig)
    else:
        raise ValueError(f"Feature importance calculation not implemented for model {model_name}.")
    
    feature_importance = feature_importance.sort_values(ascending=False)
    return mse, r2, y_pred,feature_importance  # retourne les valeurs de l'erreur quadratique et du coeff de determination

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
