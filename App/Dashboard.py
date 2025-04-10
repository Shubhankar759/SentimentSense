import pandas as pd
import re
import google.generativeai as genai
import pathlib
from pathlib import Path
import textwrap
from IPython.display import Markdown, display
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import json
import streamlit as st
import altair as alt
from wordcloud import WordCloud

# load_dotenv()

# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
# genai.configure(api_key=GOOGLE_API_KEY)

# model = genai.GenerativeModel('gemini-1.5-pro')

# # all functions needed
# def clean_text(text):
#     # Remove HTML tags
#     text = re.sub(r'<[^>]+>', '', text)

#     # Remove URLs
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

#     # Remove special characters and numbers
#     text = re.sub(r'\[.*?\]', '', text)
#     text = re.sub(r'\w*\d\w*', '', text)

#     # Convert to lowercase
#     text = text.lower()

#     # Remove punctuation
#     text = re.sub(r'[^\w\s]', '', text)

#     # Remove extra whitespace
#     text = re.sub(r'\s+', ' ', text).strip()

#     return text

# def drop_long_sentences(text):
#     if pd.isnull(text):  # Handle missing values
#         return text
#     # Count the number of words in the sentence
#     if len(text.split()) > 50:
#         return None # Replace with None to indicate cell removal
#     return text

# def graphs(final):
    
#     Field= st.selectbox(
#    'Which Field you want to display',
#     final.columns[1:])
#     # if Field in [sarcastic,ironic ,formal,toxic]:
#     #     tags=[0,1]
#     # elif Field == 'Contextual':
#     #     tags=[0,1,2,3,4,5,6,7]
#     # else Field == 'nature':
#     #     tags=[]
#     # pie_data = []
    
#     colors = ["#66c2a5","#fc8d62","#8da0cb","#e78ac3","#a6d854","#ffd92f","#e5c494","#b3b3b3",]  

#     # # Iterate through each column except 'comments'
#     # for col in final.columns:
#     #     if col != 'comments':
#     #         # Append the value counts of each column to the list
#     #         pie_data.append(final[col].value_counts())


#     # # Create subplots for each column
#     fig, ax = plt.subplots(figsize=(15, 6))
#     ax.pie(final[Field].value_counts(), labels=final[Field].value_counts().index,colors=colors,autopct='%1.1f%%', startangle=140)
#     ax.axis('equal')
#     plt.title(Field)
#     st.pyplot(fig)
    

#     # for i, data in enumerate(pie_data):
#     #     # Plot the pie chart for the current column
#     #     axes[i].pie(data, labels=data.index, autopct='%1.1f%%', startangle=90)
#     #     axes[i].set_title(final.columns[i+1])  # set title for each chart (+1 to skip comments)

#     # # Adjust layout and display the plot
#     # plt.tight_layout()
#     # plt.show()



# def Dashboard(data):
#     file_path = Path("final_predictions.csv")
    
#     if file_path.is_file():
#         final= pd.read_csv(file_path)
#         st.dataframe(final)
#         graphs(final)
    
#     else:
#         comments = data['comments']
#         cleaned = []
#         for text in comments:
#             text=str(text)
#             cleaned.append(clean_text(text))

#         data['comments'] = cleaned

#         # st.dataframe(data)
#         data['comments'] = data['comments'].apply(drop_long_sentences)

#         data['sarcastic'] = ''
#         data['nature']=''
#         data['Contextual']=''
#         data['ironic']=''
#         data['formal']=''
#         data['toxic']=''

#         json_data = testfile[['comments','sarcastic','nature','Contextual','ironic','formal','toxic']].to_json(orient='records')

#         prompt =f"""
#             Your are an expert liguist you have to classifiy the text in field comments into these field
#             sarcastic : if the text is sarcastic then fill the field with 1 if non-sarcastic then with 0
#             nature : classify text from -2 to 2 where -2 is highly negative and 2 is highly positive and 0 is neutral
#             Contextual : classify text into Supportive: 1 , Disgust:2 , Confusion:3 , Optimistic:4 , Pessimistic:5 , Humorous:6 , Mixed_feelings:7
#             ironic : if the text is ironic then fill the field with 1 if non-ironic then with 0
#             formal : if the text is formal then fill the field with 1 if non-formal then with 0
#             toxic : if the text is toxic then fill the field with 1 if non-toxic then with 0
#             if text is none then leave the other fields as it is empty
#             i have given you text in json file with respective fields please do fill them
#         ```
#         {json_data}
#         ```

#         """
    
#         response = model.generate_content(prompt)
#         json_data = response.text.replace('`','')
#         json_data = json_data.replace('\n','')
#         json_data = json_data.replace('json','')
#         data_p = json.loads(json_data)
#         final = pd.DataFrame(data_p)
#         final.to_csv('final_predictions.csv', index=False)
#         st.dataframe(final)
#         graphs(final)







def Enter_Dashboard(data):
    
    show_popup = st.button("Return Home")
    if show_popup:
        popup_container = st.container()
        with popup_container:
            st.write("it will delete comment and analyzed files")
            col1 , col2=st.columns(2)
            with col1:
                if st.button("yes"):
                    pass
            with col2:
                if st.button("no"):
                    pass
        
    
    st.dataframe(data, width=2000, height=400)
    if st.button("Refresh"):
        st.rerun()
    
    


    
    
    
    
# st.set_page_config(
# page_title="Comment Analysis",
# page_icon="🏂",
# layout="wide",
# initial_sidebar_state="expanded"
# )
# # if st.button("Back to Home"):
# #     st.session_state.Main = "Home"
    
# data= pd.read_csv("comments.csv") 

# Dashboard(data)
    

