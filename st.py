import streamlit as st
import pandas as pd
from typing import Literal
import numpy as np
import visualization 
import init_data, pre_processing
import ml 

st.title("Data Mining Project")

# Sidebar for navigation
st.sidebar.title('Navigation')
option = st.sidebar.selectbox('Choose a section:', 
    ('Initial Data Exploration', 'Data Pre-processing and Cleaning', 'Visualization', 'Clustering or Prediction'))


# Informations personnelles
for i in range(25):
    st.sidebar.markdown("")
st.sidebar.markdown("---")
st.sidebar.markdown("## SANJIVY Dorian - RESLOU Vincent")
st.sidebar.markdown("Promo 2025 - BI2")


if option == 'Initial Data Exploration':
    st.header("Initial Data Exploration")
    # Upload the CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    # Variables to hold separator and header choice
    sep = None
    header = None

    st.set_option('deprecation.showPyplotGlobalUse', False)



    # Lorsque que la file présente
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

# Lorsque que la file est charger
if 'data' in st.session_state:
    if option == 'Data Pre-processing and Cleaning':
        st.header("Data Pre-processing and Cleaning")

        st.subheader("Managing missing values:")
        # User options for handling missing data
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
        #add a selection box to choose the normalization method
        normalization_method = st.selectbox("Choose the normalization method",['Min-Max','Z-standardization','Robust'])
        #add a button to normalize the data
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
        #add a button to display the histograms
        if st.button("Display histograms"):
            visualization.histograms(st.session_state['data'])
        #add a button to display the box plots
        if st.button("Display box plots"):
            visualization.box_plots(st.session_state['data'])
        # ADD +1 VISUALISATION (au moins)


    if option == 'Clustering or Prediction':
        st.header("Clustering or Prediction (Visualization and Evaluation)")
        st.subheader("Clustering:")
        #add a selection box to choose the clustering algorithm
        algorithm = st.selectbox("Choose the clustering algorithm",['kmeans','dbscan'])
        if algorithm == 'dbscan':
            eps = st.number_input("Enter the maximum distance between two samples",min_value=1.0)
            min_samples = st.number_input("Enter the number of samples in a neighborhood for a point to be considered as a core point",min_value=1)
        
        elif algorithm == 'kmeans':
            #add a selection box to choose the number of clusters
            cluster_number =st.selectbox("Choose the number of clusters",[1,2,3,4,5])

        target_column = st.selectbox("Choose the x",st.session_state['data'].columns)
        target_column2 = st.selectbox("Choose the y",st.session_state['data'].columns)
        #add a button to perform clustering
        if st.button("Perform clustering"):
            if algorithm == 'kmeans':
                st.session_state['data'], test_score = ml.perform_clustering(st.session_state['data'],algorithm,n_clusters=cluster_number)
                #fait une figure pour afficher les clusters
                visualization.plot_clusters(st.session_state['data'],target_column,target_column2)
                st.write(f"Test score: {test_score}")
            elif algorithm == 'dbscan':
                st.session_state['data'], test_score = ml.perform_clustering(st.session_state['data'],algorithm,eps=eps,min_samples=min_samples)
                visualization.plot_clusters(st.session_state['data'], target_column, target_column2)
        #Pas ultra convaincu a revoir + de stats


        st.subheader("Prediction:")
        #add a selection box to choose the prediction algorithm
        algorithm = st.selectbox("Choose the prediction algorithm",['linear_regression','decision_tree'])
        #add a button to perform prediction
        target_column = st.selectbox("Choose the target column",st.session_state['data'].columns)
        if st.button("Perform prediction"):
            if algorithm == 'linear_regression':
                model, scores = ml.perform_prediction(st.session_state['data'],target_column,algorithm)
            elif algorithm == 'decision_tree':
                model, scores = ml.perform_prediction(st.session_state['data'],target_column,algorithm)
            st.write(st.session_state['data'])
            st.write(scores)
        
        #Jsp ajouter graph + stats + test