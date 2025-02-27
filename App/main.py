import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from Login import login
from Home import fileuploader

login()
if st.session_state['authentication_status']:
    st.title('SentimentSense')
    fileuploader()   
elif st.session_state['authentication_status'] is False:
    st.error('Username/password is incorrect')
elif st.session_state['authentication_status'] is None:
    st.warning('Please enter your username and password')
    



