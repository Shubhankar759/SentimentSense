import streamlit as st
import pandas as pd


def fileuploader():
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])
    
    if uploaded_file is None:
        st.error("Upload a file")
    else:
        df = pd.read_csv(uploaded_file)
        st.write(df)
        
        