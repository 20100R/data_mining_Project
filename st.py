#Streamlit
import streamlit as st
import pandas as pd
from typing import Literal
import numpy as np
import visualization 
import init_data, pre_processing
#display the title
st.title("Data Mining Project")

#do a selection box to choose the sepertor and the header
sep = st.selectbox("Choose the separator",[';',',','tab'])
header = st.selectbox("Choose the header",[0,1])

#add a button to load the data

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
data=init_data.load_csv(uploaded_file,header=header,sep=sep)
init_data.display_description(data)
init_data.statistical_summary(data)

#add a button to delete the empty rows and columns
if st.button("Delete empty rows and columns"):
    data = pre_processing.delete_empty_row_col(data)
    st.write(data)
#add a button to replace the missing values
method = st.selectbox("Choose the method to replace the missing values",['mean','median','mode'])
if st.button("Replace missing values"):
    data = pre_processing.replace_missing_values(data,method)
    st.write(data)
#add a selection box to choose the normalization method
normalization_method = st.selectbox("Choose the normalization method",['Min-Max','Z-standardization'])
#add a button to normalize the data
if st.button("Normalize the data"):
    if normalization_method == 'Min-Max':
        data = pre_processing.normalize_min_max(data)
    elif normalization_method == 'Z-standardization':
        data = pre_processing.normalize_z_standardization(data)
    st.write(data)

#add a button to display the histograms
if st.button("Display histograms"):
    st.write(data)
    visualization.histograms(data)
#add a button to display the box plots
if st.button("Display box plots"):
    st.write(data)
    visualization.box_plots(data)