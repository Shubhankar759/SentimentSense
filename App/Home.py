import streamlit as st
import pandas as pd
import httpx

def reddit_comment_extractor(post_url,sort=None,limit=50):
    url=post_url+'comments.json'
    
    dataset=[]
    
    params={
        'sort':sort
    }
    
    response = httpx.get(url,params=params)
    
    print(f'fetching "{response.url}"...')
    
    if response.status_code != 200:
        raise Exception('Failed to fetch data')

    json_data = response.json()

    dataset.extend(child['data']['body'].replace(',', '').replace('.', '')
                   for child in json_data[1]['data']['children'] 
                   if 'body' in child['data'])

    return pd.DataFrame(dataset, columns=['comments'])



def home():
    logo , mids , abt = st.columns([1, 5, 1]) 
    
    with abt:
        if st.button('About us'):
            st.session_state.Main="About"
    

    # Logout button
    with logo:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.username = None
            st.session_state.page = "login"
            st.rerun()
    
    st.title("Home Page")
    
    tab1, tab2 = st.tabs(["Upload File", "Social Media API"])
    
    *_,startana =st.columns([1, 3, 1])
    
    with tab1:

        uploaded_file = st.file_uploader("Choose a file", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            df.to_csv("comments.csv", index=False)

        if st.button("Start Analysis"):
            st.session_state.Main = "Dashboard"   
        

        
    with tab2:
        
        option = st.selectbox(
            "Social Media",
            [None,"Reddit"],
            format_func=lambda x: "Select a Social Media Platform" if x is None else x
        )

        user_input = st.text_input("Enter your Link of Social Media")
        
        num_comments = st.slider(label="Choose a value:",min_value=20,max_value=100,value=50)

        if st.button("Get Comments & Start Analysis"):
            if option =='Reddit': 
                
                df = reddit_comment_extractor(user_input)
                df.to_csv("comments.csv", index=False)
                if df.empty: st.error("Insert URL of Post")
            
            st.session_state.Main = "Dashboard"  
    

