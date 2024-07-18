import streamlit as st
import pandas as pd
import init_data, pre_processing, visualization, ml
from sklearn.model_selection import train_test_split
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
    # Upload the CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    # Variables to hold separator and header choice
    sep = None
    header = None

    st.set_option('deprecation.showPyplotGlobalUse', False)



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
        #Box for the visalization
        if st.button("Display histograms"):
            visualization.histograms(st.session_state['data'])
        if st.button("Display box plots"):
            visualization.box_plots(st.session_state['data'])
        if st.button("Display heatmap"):
            visualization.correlation_heatmap(st.session_state['data'])


    if option == 'Clustering or Prediction':
        st.header("Clustering or Prediction (Visualization and Evaluation)")
        st.subheader("Clustering:")
        #Selection box to choose the clustering algorithm
        algorithm = st.selectbox("Choose the clustering algorithm",['kmeans','dbscan'])
        if algorithm == 'dbscan':
            eps = st.number_input("Enter the maximum distance between two samples",min_value=1.0)
            min_samples = st.number_input("Enter the number of samples in a neighborhood for a point to be considered as a core point",min_value=1)
        
        elif algorithm == 'kmeans':
            #Selection for the number of clusters
            cluster_number =st.selectbox("Choose the number of clusters",[1,2,3,4,5])

        target_column = st.selectbox("Choose the x",st.session_state['data'].columns)
        target_column2 = st.selectbox("Choose the y",st.session_state['data'].columns)

        if st.button("Perform clustering"):
            if algorithm == 'kmeans':
                st.session_state['data'], test_score = ml.perform_clustering(st.session_state['data'],algorithm,n_clusters=cluster_number)
                #fait une figure pour afficher les clusters ????
                visualization.plot_clusters(st.session_state['data'],target_column,target_column2)
                st.write(f"Test score: {test_score}")
            elif algorithm == 'dbscan':
                st.session_state['data'], test_score = ml.perform_clustering(st.session_state['data'],algorithm,eps=eps,min_samples=min_samples)
                visualization.plot_clusters(st.session_state['data'], target_column, target_column2)
        #Pas ultra convaincu a revoir + de stats


        st.subheader("Prediction:")
        #Selection box to choose the prediction algorithm
        target_column = st.selectbox("Choose the target column",st.session_state['data'].columns)
        if st.button("Test models"):
            resultats=ml.perform_prediction(st.session_state['data'],target_column)
            visualization.plot_prediction(resultats)
        
        models = st.selectbox("Choose the model",['Regression Linear','Regression Ridge','Regression Lasso','K-Nearest Neighbors','Decision Tree'])
        if st.button("Prediction"):
            df=st.session_state['data']
            x=df.drop(target_column, axis=1).values
            y=df[target_column].values
            X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
            mse, r2, y_pred =ml.predict(models,st.session_state['data'],target_column)
            #afficher les resultats
            st.write(f"{models} :MSE {mse}, R2 {r2}")
            visualization.plot_predict(y_test, y_pred)