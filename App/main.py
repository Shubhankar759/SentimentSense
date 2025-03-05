import streamlit as st
from Login import login
from signup import signup
from forgot_password import forgot_password
from forgot_username import forgot_username
from Home import home
from About import about_us

# Initialize session state if not set
if "page" not in st.session_state:
    st.session_state["page"] = "login"

# Route pages based on session state
if st.session_state["page"] == "login":
    login()
elif st.session_state["page"] == "signup":
    signup()
elif st.session_state["page"] == "forgot_password":
    forgot_password()
elif st.session_state["page"] == "forgot_username":
    forgot_username()
elif st.session_state["page"] == "Home":
    home()
elif st.session_state["page"] == "About":
    about_us()
