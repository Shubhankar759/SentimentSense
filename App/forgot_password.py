import streamlit as st

def forgot_password():
    st.title("Forgot Password")

    email = st.text_input("Enter your registered email")

    if st.button("Reset Password"):
        st.success("A password reset link has been sent to your email!")

    if st.button("Back to Login"):
        st.session_state["page"] = "login"
