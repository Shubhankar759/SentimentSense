import streamlit as st
from Home import home 
from About import about_us
from Authetication import AuthenticationSystem , StreamlitAuthUI
from Dashboard import Enter_Dashboard
import os
from pathlib import Path
import pandas as pd

def Inapp(): 
    if st.session_state.Main == "Home":
        st.set_page_config(
            page_title="Home",
            page_icon="./SentimentSense.svg",
            layout="centered",
            initial_sidebar_state="expanded"
        )
        home()
    elif st.session_state.Main == "About":
        st.set_page_config(
            page_title="Home",
            page_icon="./SentimentSense.svg",
            layout="centered",
            initial_sidebar_state="expanded"
        )
        about_us()
        
    elif st.session_state.Main == "Dashboard":
        st.set_page_config(
            page_title="Home",
            page_icon="./SentimentSense.svg",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        file_path = Path("comments.csv")
        if file_path.is_file():
            data= pd.read_csv(file_path) 
            Enter_Dashboard(data)
        else:
            st.error("File not found")
        
        
       
    


# Initialize session state if not set
if "page" not in st.session_state:
    st.session_state.page = "login"

# Route pages based on session state
    
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "Main" not in st.session_state:
    st.session_state.Main = "Home"

auth_system = AuthenticationSystem()
auth_ui = StreamlitAuthUI(auth_system)
query_params = st.query_params
token = query_params.get("token", [None])[0]



if st.session_state.logged_in:
    Inapp()
else:
    if token and st.session_state.page == "login":
        auth_ui.reset_password_page(token)
    elif st.session_state.page == "signup":
        st.set_page_config(
            page_title="Home",
            page_icon="./SentimentSense.svg",
            layout="centered",
            initial_sidebar_state="expanded"
        )
        auth_ui.signup_page() 
    else:
        st.set_page_config(
            page_title="Home",
            page_icon="./SentimentSense.svg",
            layout="centered",
            initial_sidebar_state="expanded"
        )
        auth_ui.login_page()

