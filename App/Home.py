import streamlit as st
import pandas as pd

def home():
    st.title("Home Page")

    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(df)

    # Button to navigate to About Us
    if st.button("About Us"):
        st.session_state["page"] = "About"

    # Logout button
    if st.button("Logout"):
        st.session_state["page"] = "login"
