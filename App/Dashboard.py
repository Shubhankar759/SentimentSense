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

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-1.5-pro')

# all functions needed
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove special characters and numbers
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\w*\d\w*', '', text)

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def graphs(final):
    label_counts = final['predicte'].value_counts().reset_index()
    label_counts.columns = ['label', 'count']
    final['body']=str(final['body'])
    text = " ".join(final['body'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Create a pie chart with Altair
    chart = alt.Chart(label_counts).mark_arc().encode(
        theta=alt.Theta(field='count', type='quantitative'),
        color=alt.Color(field='label', type='nominal'),
        tooltip=['label', 'count']
    ).properties(title='Distribution of Sarcastic and Non-sarcastic Comments')

    # Streamlit app
    st.title("Sarcasm Detection Pie Chart")
    st.altair_chart(chart, use_container_width=True)
    
    st.title("Word Cloud")
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)


def Dashboard(data):
    data=data.iloc[:100, :4]
    file_path = Path("final_predictions.csv")
    
    
    comments = data['body']
    cleaned = []
    for text in comments:
        text=str(text)
        cleaned.append(clean_text(text))

    data['body'] = cleaned

    data.head()
    
    data['predicte'] = ''
    
    json_data = data[['body','predicte']].to_json(orient='records')
    
    prompt =f"""
    Your are an expert liguist , who have good understanding of sentiments and understand sarcasm you are good at classifying social media comments into sarcastic and non-sarcastic labels.
    help me classify the comments into: sarcastic (label=1) and non-sarcastic(label=0) labels.
    data is given in json file which consist of two attributes body , predicte.
    'body' is the main comment to be analysied
    'predicte' is to be labelled as sarcastic or non-sarcastic
    your task is to update predicted lables under 'predicte' in json code
    dont make any changes to json code format, please.
    ```
    {json_data}
    ```

    """
    if file_path.is_file():
        final= pd.read_csv(file_path)
        st.dataframe(final)
        graphs(final)
    else:
        response = model.generate_content(prompt)
        json_data = response.text.replace('`','')
        json_data = json_data.replace('\n','')
        json_data = json_data.replace('json','')

        data_p = json.loads(json_data)
        final = pd.DataFrame(data_p)
        final.to_csv('final_predictions.csv', index=False)
        st.dataframe(final)

        graphs(final)



    
st.set_page_config(
    page_title="US Population Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")



    
if st.button("Click Me"):
    data = pd.read_csv("C:/Users/shubh/Downloads/archive/Comments_reddit.csv")  
    Dashboard(data)
    

    



