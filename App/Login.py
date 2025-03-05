import os
import streamlit as st
import yaml
from yaml.loader import SafeLoader


def Options_login():
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


def login():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    if not os.path.exists(config_path):
        st.error("Config file not found. Please check the path.")
        return

    # Load credentials from config.yaml
    with open(config_path) as file:
        config = yaml.load(file, Loader=SafeLoader)

    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        # Verify credentials manually (Plain-Text)
        if username in config["credentials"]["usernames"] and password == config["credentials"]["usernames"][username][
            "password"]:
            st.session_state["page"] = "Home"
            st.session_state["name"] = config["credentials"]["usernames"][username]["first_name"]
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Incorrect username or password")

    Options_login()
