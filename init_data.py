import pandas as pd
import streamlit as st
import io

def load_data(data_file, names_file, sep, header):
    # Read .names file to get column names if provided
    if names_file is not None:
        column_names = [line.split(':')[0].strip() for line in names_file.getvalue().decode('utf-8').split('\n') if line.strip() and ':' in line]
    else:
        column_names = None  # Fallback if no names file is provided

    # Convert 'tab' to '\t' if selected
    sep = '\t' if sep == 'tab' else sep

    # Load the .data file
    if data_file is not None:
        if column_names:
            data = pd.read_csv(data_file, header=None, names=column_names, sep=sep)
        else:
            data = pd.read_csv(data_file, header=header, sep=sep)
    else:
        data = None

    return data

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

