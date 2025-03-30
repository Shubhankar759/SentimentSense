import streamlit as st
import pandas as pd





def home():
    st.title("Home Page")
    
    tab1, tab2 = st.tabs(["Upload File", "Social Media API"])
    
    with tab1:
        # File uploader
        uploaded_file = st.file_uploader("Choose a file", type=["csv"])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write(df)
        
        api_req=False
        
    # st.markdown(
    # "<h1 style='text-align: center;'>Or</h1>",
    # unsafe_allow_html=True
    # )
    
    with tab2:
        option = st.selectbox(
            "Social Media",
            ["Youtube"]
        )

        user_input = st.text_input("Enter your Link of Social Media")
        
        num_comments = st.slider(label="Choose a value:",min_value=20,max_value=100,value=50)
        
        api_req=True
        

    
    # Button to navigate to About Us
   
    # if st.button("About Us"):
    #     st.session_state["page"] = "About"
    
    if st.button("Start Analysis"):
        pass
    
    if st.button('About us'):
        st.session_state.Main="About"
    

    # Logout button

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.username = None
        st.session_state.page = "login"
        st.rerun()
