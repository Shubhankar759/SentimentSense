import os
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

def login():
    # Define the correct path for config.yaml
    config_path = 'App\config.yaml'


    # Check if the file exists before opening
    if not os.path.exists(config_path):
        st.error(f"Config file not found: {config_path}")
        return

    with open(config_path) as file:
        config = yaml.load(file, Loader=SafeLoader)


    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )


    st.title("Login")


    try:
        authenticator.login()
    except Exception as e:
        st.error(e)


    if st.session_state.get('authentication_status'):
        authenticator.logout()
        st.write(f'Welcome *{st.session_state["name"]}*')
    elif st.session_state.get('authentication_status') is False:
        st.error('Username/password is incorrect')
    elif st.session_state.get('authentication_status') is None:
        st.warning('Please enter your username and password')

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Sign Up"):
            st.session_state["page"] = "signup"
    with col2:
        if st.button("Forgot Password?"):
            st.session_state["page"] = "forgot_password"
    with col3:
        if st.button("Forgot Username?"):
            st.session_state["page"] = "forgot_username"
