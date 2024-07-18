import pandas as pd
import streamlit as st
import io

# Function to detect separator
def detect_separator(sample: str):
    separators = [';', ',', '\t']  
    counts = {sep: sample.count(sep) for sep in separators}
    return max(counts, key=counts.get) 

# Function to display lot of information about the dataset
def display_description(data:pd.DataFrame):
    st.write("Data first 5 value:")
    st.write(data.head())
    st.write("Data last 5 value:")
    st.write(data.tail())
    st.write("Describe:")
    st.write(data.describe())
    st.write("Info:")
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.markdown('```plaintext\n' + s + '\n```')
    st.write("Missing Data:")
    st.write(data.isnull().sum())

