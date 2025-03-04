import streamlit as st

def signup():
    st.title("Sign Up")

    new_username = st.text_input("Choose a Username")
    new_email = st.text_input("Enter Email")
    new_password = st.text_input("Choose a Password", type="password")

    if st.button("Register"):
        st.success("Account created successfully!")
        st.session_state["page"] = "login"

    if st.button("Back to Login"):
        st.session_state["page"] = "login"