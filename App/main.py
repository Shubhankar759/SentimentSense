import streamlit as st
from Login import login
from Home import fileuploader
from signup import signup
from forgot_password import forgot_password
from forgot_username import forgot_username
from Home import fileuploader


login()

    



if "page" not in st.session_state:
    st.session_state["page"] = "login"

if st.session_state["page"] == "login":
    login()
elif st.session_state["page"] == "signup":
    signup()
elif st.session_state["page"] == "forgot_password":
    forgot_password()
elif st.session_state["page"] == "forgot_username":
    forgot_username()
