import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import init_data, pre_processing, visualization, ml

st.title("Data Mining Project")

# Sidebar for navigation
st.sidebar.title('Navigation')
option = st.sidebar.selectbox('Choose a section:', 
    ('Initial Data Exploration', 'Data Pre-processing and Cleaning', 'Visualization', 'Clustering or Prediction'))


# Personal Information
for i in range(25):
    st.sidebar.markdown("")
st.sidebar.markdown("---")
st.sidebar.markdown("## SANJIVY Dorian - RESLOU Vincent")
st.sidebar.markdown("Promo 2025 - BI2")


if option == 'Initial Data Exploration':
    st.header("Initial Data Exploration")

    # Choose file type
    file_type = st.radio("Select the type of file to upload:", ('CSV', '.data and .names'))

    # Variables to hold separator and header choice
    sep = None
    header = None

    st.set_option('deprecation.showPyplotGlobalUse', False)

    if file_type == 'CSV':
        # Upload the CSV file
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        # When the file is present
        if uploaded_file is not None:
            # Read the first line to detect the separator
            sample = uploaded_file.getvalue().decode('utf-8').split('\n')[0]
            detected_sep = init_data.detect_separator(sample)
            
            # Let user choose or confirm the detected separator
            sep = st.selectbox("Choose the separator", options=[f'Auto-detect: {detected_sep}',';', ',', 'tab'])
            
            # If auto-detect is selected, use the detected separator, convert 'tab' to '\t'
            if sep.startswith('Auto-detect'):
                sep = detected_sep if detected_sep != 'tab' else '\t'
            
            # Let user select the header row
            header = st.selectbox("Choose the header", [0, 1, 'None'])
            header = None if header == 'None' else header

            # Load the data using the selected separator and header
            data = pd.read_csv(uploaded_file, header=header, sep=sep)

            # Display Data
            init_data.display_description(data)

            if 'data' not in st.session_state:
                st.session_state['data'] = data
                st.write("Data loaded successfully")
            else:
                st.write("Data loaded successfully")

    elif file_type == '.data and .names':
        # File uploaders for .data and .names files
        data_file = st.file_uploader("Upload .data file", type=['data'])
        names_file = st.file_uploader("Upload .names file", type=['names'])
        
        if data_file and names_file:

            # Read the first line to detect the separator
            first_line = data_file.getvalue().decode('utf-8').split('\n')[0]
            detected_sep = init_data.detect_separator(first_line)

            sep = st.selectbox("Choose the separator", options=[f'Auto-detect: {detected_sep}', ';', ',', 'tab'])

            if sep.startswith('Auto-detect'):
                sep = detected_sep if detected_sep != 'tab' else '\t'

            # Resetting file pointer after reading
            data_file.seek(0)


            data = init_data.load_data(data_file, names_file, sep, header)

            # Display Data
            init_data.display_description(data)

            if 'data' not in st.session_state:
                st.session_state['data'] = data
                st.write("Data loaded successfully")
            else:
                st.write("Data loaded successfully")


# When the file is loaded
if 'data' in st.session_state:
    if option == 'Data Pre-processing and Cleaning':
        st.header("Data Pre-processing and Cleaning")

        st.subheader("Managing missing values:")

        # Options for handling missing data
        options = ["Delete empty rows and columns", "Replace missing values (mean)", "Replace missing values (median)", "Replace missing values (mode)","Replace missing values (knn imputation)"]
        choice = st.selectbox("Select how to handle missing data:", options)

        if st.button("Apply"):
            if choice == "Delete empty rows and columns":
                st.session_state['data'] = pre_processing.delete_missing_row_col(st.session_state['data'])
            elif choice == "Replace missing values (mean)":
                st.session_state['data'] = pre_processing.replace_missing_values(st.session_state['data'], 'mean')
            elif choice == "Replace missing values (median)":
                st.session_state['data'] = pre_processing.replace_missing_values(st.session_state['data'], 'median')
            elif choice == "Replace missing values (mode)":
                st.session_state['data'] = pre_processing.replace_missing_values(st.session_state['data'], 'mode')
            elif choice == "Replace missing values (knn imputation)":
                st.session_state['data'] = pre_processing.replace_missing_values(st.session_state['data'], 'knn')
            
            st.write("Updated Data:")
            st.write(st.session_state['data'])



        st.subheader("Data normalization:")
        #Selection box to choose the normalization method
        normalization_method = st.selectbox("Choose the normalization method",['Min-Max','Z-standardization','Robust'])
        if st.button("Normalize the data"):

            if normalization_method == 'Min-Max':
                st.session_state['data'] = pre_processing.normalize_min_max(st.session_state['data'])

            elif normalization_method == 'Z-standardization':
                st.session_state['data'] = pre_processing.normalize_z_standardization(st.session_state['data'])
            
            elif normalization_method == 'Robust':
                st.session_state['data'] = pre_processing.normalize_robust(st.session_state['data'])
            
            st.write("Normalize Data:")
            st.write(st.session_state['data'])


    if option == 'Visualization':
        st.header("Visualization")

        if st.button("Display histograms"):
            visualization.histograms(st.session_state['data'])

        if st.button("Display box plots"):
            visualization.box_plots(st.session_state['data'])

        if st.button("Display heatmap"):
            visualization.correlation_heatmap(st.session_state['data'])


    if option == 'Clustering or Prediction':
        st.header("Clustering or Prediction (Visualization and Evaluation)")
        st.subheader("Clustering:")

        # Filter to only numeric columns
        data = st.session_state['data']
        numeric_columns = data.select_dtypes(include=[np.number]).columns

        # Dropdown to select features
        feature_1 = st.selectbox("Select Feature 1", numeric_columns, index=0)
        feature_2 = st.selectbox("Select Feature 2", numeric_columns, index=1)
        selected_data = data[[feature_1, feature_2]]
        
        # Selection box to choose the clustering algorithm
        algorithm = st.selectbox("Choose the clustering algorithm", ['kmeans', 'dbscan', 'spectral'])
        
        # Parameters for K-Means
        if algorithm == 'kmeans':
            n_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=3)
        # Parameters for DBSCAN
        elif algorithm == 'dbscan':
            eps = st.slider("EPS (epsilon)", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
            min_samples = st.slider("Minimum samples", min_value=1, max_value=10, value=5)
        elif algorithm == 'spectral':
            n_clusters = st.slider("Number of clusters for Spectral Clustering", min_value=2, max_value=10, value=3)

        
        if st.button("Perform clustering"):
            
            # Perform clustering
            if algorithm == 'kmeans':
                labels, centers = ml.perform_clustering(selected_data, algorithm, n_clusters=n_clusters)
                visualization.plot_clusters(selected_data, labels, [feature_1, feature_2], centers=centers)
                stats_df = ml.compute_cluster_stats(selected_data, labels, centers)

            elif algorithm == 'dbscan':
                labels = ml.perform_clustering(selected_data, algorithm, eps=eps, min_samples=min_samples)
                visualization.plot_clusters(selected_data, labels, [feature_1, feature_2])
                stats_df = ml.compute_cluster_stats(selected_data, labels)

            elif algorithm == 'spectral':
                labels = ml.perform_clustering(selected_data, algorithm, n_clusters=n_clusters)
                visualization.plot_clusters(selected_data, labels, [feature_1, feature_2])
                stats_df = ml.compute_cluster_stats(selected_data, labels)
            
            st.write("Cluster Statistics:")
            st.dataframe(stats_df)
            

        st.subheader("Prediction:")
        #Selection box to choose the target
        target_column = st.selectbox("Choose the target column",st.session_state['data'].columns)
        if st.button("Test models"):
            resultats=ml.perform_prediction(st.session_state['data'],target_column)
            visualization.plot_prediction(resultats)

        #Selection box to choose the prediction algorithm
        models = st.selectbox("Choose the model",['Regression Linear','Regression Ridge','Regression Lasso','K-Nearest Neighbors','Decision Tree'])
        if st.button("Prediction"):
            df=st.session_state['data']
            x=df.drop(target_column, axis=1).values
            y=df[target_column].values
            X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
            mse, r2, y_pred,feature_importance =ml.predict(models,st.session_state['data'],target_column)
            
            #print the results 
            st.write(f"{models} :MSE {mse}, R2 {r2}")
            visualization.plot_predict(y_test, y_pred)
            visualization.plot_feature_importance(feature_importance,df.columns.drop(target_column))