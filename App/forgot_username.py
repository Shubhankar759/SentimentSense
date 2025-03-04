import streamlit as st

def forgot_username():
    st.title("Forgot Username")

    email = st.text_input("Enter your registered email")

    if st.button("Retrieve Username"):
        st.success("Your username has been sent to your email!")

    if st.button("Back to Login"):
        st.session_state["page"] = "login"