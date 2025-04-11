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
import requests
from io import StringIO
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
import re
import seaborn as sns


load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-1.5-pro')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# all functions needed
@st.cache_data
def preprocessing(text):
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


@st.cache_data
def clean_text(text):
    if isinstance(text, str):
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words and len(word) > 2])
    return text

@st.cache_data
def drop_long_sentences(text):
    if pd.isnull(text):  # Handle missing values
        return text
    # Count the number of words in the sentence
    if len(text.split()) > 50:
        return None # Replace with None to indicate cell removal
    return text

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



def graph(df):
    df['clean_comments'] = df['comments'].apply(clean_text)

    # Title
    st.title("SentimentSense Analysis")

    # Sidebar with attribute selection

    with st.sidebar:

        st.sidebar.header("Select Attributes to Analyze")
        attributes = ['sarcastic', 'nature', 'Contextual', 'ironic', 'formal', 'toxic']
        selected_attributes = []

        for attr in attributes:
            if st.sidebar.checkbox(attr, value=True):
                selected_attributes.append(attr)

        st.markdown("---")
        st.markdown("### Quick Analysis")
        quick_text = st.text_area("Enter text for quick analysis", height=100)

        if quick_text:
            with st.spinner("Analyzing..."):
                
                
                prompt=f"""
                text:{quick_text}
                give me a sentiment analysis of give text and confidence between 0.0 and 1.0 on the basis of attribute
                'sarcastic','positive','funny','ironic','sensible','toxic' in json file only the value with respective attribute
                """
                quick_result = model.generate_content(prompt)
                scores = json.loads(quick_result.text.replace('json','').replace("`",'').replace('\n',''))

                st.title("Text Trait Analysis")

                # Display each trait as a heading + progress bar
                for trait, value in scores.items():
                    st.subheader(trait.capitalize())
                    st.progress(value)



    # Main content
    if not selected_attributes:
        st.warning("Please select at least one attribute to analyze.")
    else:
        # Overview section
        st.header("Dataset Overview")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Data Sample")
            st.dataframe(df[['comments'] + attributes],width=800, height=315)

        with col2:
            st.subheader("Data Statistics")
            st.dataframe(df[attributes].describe(),width=800, height=315)

        # Pie charts section
        st.header("Attribute Distribution (Pie Charts)")

        # Create pie charts in rows of 3
        cols = st.columns(min(3, len(selected_attributes)))

        for i, attr in enumerate(selected_attributes):
            col_idx = i % 3
            with cols[col_idx]:
                st.subheader(f"{attr} Distribution")

                # Count values
                value_counts = df[attr].value_counts()

                # Create pie chart
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                st.pyplot(fig)

                # Display counts as a table
                st.dataframe(pd.DataFrame({
                    'Value': value_counts.index,
                    'Count': value_counts.values,
                    'Percentage': (value_counts.values / value_counts.sum() * 100).round(2)
                }))

        # Word clouds for comments filtered by attribute values
        st.header("Word Clouds for Comments by Attribute Values")

        for attr in selected_attributes:
            st.subheader(f"Word Clouds for Comments by {attr} Values")

            # Get unique values for this attribute
            unique_values = df[attr].unique()

            # Create columns for word clouds
            cols = st.columns(min(3, len(unique_values)))

            for i, value in enumerate(unique_values):
                col_idx = i % 3
                with cols[col_idx]:
                    # Filter comments by attribute value
                    filtered_comments = df[df[attr] == value]['clean_comments'].dropna()

                    if not filtered_comments.empty:
                        # Combine filtered comments
                        combined_text = ' '.join(filtered_comments)

                        if combined_text.strip():
                            st.write(f"### {attr} = {value}")

                            # Create word cloud
                            wc = WordCloud(width=800, height=400, 
                                          background_color='white', 
                                          max_words=100,
                                          collocations=False)
                            wc.generate(combined_text)

                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.imshow(wc, interpolation='bilinear')
                            ax.axis('off')
                            st.pyplot(fig)

                            # Show top words
                            word_counts = Counter(combined_text.split())
                            top_words = pd.DataFrame(word_counts.most_common(10), columns=['Word', 'Count'])
                            st.write("Top 10 words:")
                            st.dataframe(top_words)
                        else:
                            st.write(f"No meaningful text found for {attr} = {value}")
                    else:
                        st.write(f"No comments found for {attr} = {value}")

        # Word cloud for all comments
        st.header("Word Cloud for All Comments")

        # Combine all cleaned comments
        all_comments = ' '.join(df['clean_comments'].dropna())

        # Create word cloud
        if all_comments:
            wc = WordCloud(width=1000, height=500, 
                          background_color='white', 
                          max_words=200, 
                          collocations=False)
            wc.generate(all_comments)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

            # Show top words
            word_counts = Counter(all_comments.split())
            top_words = pd.DataFrame(word_counts.most_common(20), columns=['Word', 'Count'])
            st.write("Top 20 words in all comments:")
            st.dataframe(top_words)
        else:
            st.write("No comments found in the dataset.")

        # Additional visualizations
        st.header("Additional Visualizations")

        # Correlation heatmap
        st.subheader("Correlation Heatmap")

        # Convert columns to numeric if possible
        numeric_df = df[attributes].copy()
        for col in numeric_df.columns:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')

        # Create correlation matrix
        corr = numeric_df.corr()

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Bar charts for attribute counts

        for attr in selected_attributes:
            st.write(f"### {attr} Value Counts")
            
            st.write(df[attr].value_counts())

            # Count values
            value_counts = df[attr][df[attr].isin([0, 1])].value_counts().sort_index()

            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
            ax.set_title(f"{attr} Value Counts")
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")

            # Rotate x-axis labels if there are many values
            if len(value_counts) > 5:
                plt.xticks(rotation=45)

            st.pyplot(fig)

        # Pairwise relationships
        if len(selected_attributes) >= 2:
            st.subheader("Pairwise Relationships")

            # Select first two attributes for demonstration
            attr1, attr2 = selected_attributes[:2]

            # Create crosstab
            cross_tab = pd.crosstab(df[attr1], df[attr2])

            # Plot heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cross_tab, annot=True, cmap='viridis', fmt='d', ax=ax)
            ax.set_title(f"Relationship between {attr1} and {attr2}")
            st.pyplot(fig)

            # Create stacked bar chart
            st.write(f"### Stacked Bar Chart: {attr1} vs {attr2}")

            # Normalize the crosstab
            cross_tab_norm = cross_tab.div(cross_tab.sum(axis=1), axis=0)

            # Plot stacked bar chart
            fig, ax = plt.subplots(figsize=(12, 6))
            cross_tab_norm.plot(kind='bar', stacked=True, ax=ax)
            ax.set_title(f"Proportion of {attr2} values for each {attr1} value")
            ax.set_xlabel(attr1)
            ax.set_ylabel(f"Proportion of {attr2}")
            ax.legend(title=attr2)

            st.pyplot(fig)

        # NEW VISUALIZATIONS

        # 1. Interactive Sunburst Chart
        st.header("Interactive Sunburst Chart")

        if len(selected_attributes) >= 2:
            # Select the first two attributes for the sunburst
            attr1, attr2 = selected_attributes[:2]

            # Count combinations
            sunburst_data = df.groupby([attr1, attr2]).size().reset_index(name='count')

            # Create sunburst chart
            fig = px.sunburst(
                sunburst_data, 
                path=[attr1, attr2], 
                values='count',
                title=f'Sunburst Chart of {attr1} and {attr2}'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Select at least two attributes to display the sunburst chart.")

        # 2. Comment Length Analysis
        st.header("Comment Length Analysis")

        # Calculate comment length
        df['comment_length'] = df['comments'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)

        # Create histogram
        fig = px.histogram(
            df, 
            x='comment_length',
            nbins=50,
            title='Distribution of Comment Lengths',
            labels={'comment_length': 'Comment Length (characters)', 'count': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # 3. Comment Length by Attribute
        st.header("Comment Length by Attribute Values")

        for attr in selected_attributes:
            # Group by attribute and calculate mean comment length
            grouped_data = df.groupby(attr)['comment_length'].agg(['mean', 'median', 'count']).reset_index()

            # Create bar chart
            fig = px.bar(
                grouped_data,
                x=attr,
                y='mean',
                error_y='median',
                title=f'Average Comment Length by {attr} Value',
                labels={attr: f'{attr} Value', 'mean': 'Average Length (characters)'},
                text='count'
            )
            fig.update_traces(texttemplate='%{text} comments', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

        # 4. Text Clustering Visualization
        st.header("Text Clustering Visualization")

        # Check if we have enough non-empty comments
        if df['clean_comments'].dropna().shape[0] > 10:
            # Create a sample for visualization (for performance)
            sample_size = min(1000, df.shape[0])
            df_sample = df.dropna(subset=['clean_comments']).sample(sample_size, random_state=42,replace=True)

            # Vectorize the text
            vectorizer = CountVectorizer(max_features=1000)
            X = vectorizer.fit_transform(df_sample['clean_comments'])

            # Apply PCA for dimensionality reduction
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(X.toarray())

            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(X)

            # Create a DataFrame for visualization
            cluster_df = pd.DataFrame({
                'x': pca_result[:, 0],
                'y': pca_result[:, 1],
                'cluster': clusters
            })

            # Add a sample attribute for coloring
            if selected_attributes:
                cluster_df[selected_attributes[0]] = df_sample[selected_attributes[0]].values

            # Create scatter plot
            fig = px.scatter(
                cluster_df,
                x='x',
                y='y',
                color='cluster',
                hover_data=[selected_attributes[0]] if selected_attributes else None,
                title='Text Clustering based on Comment Content',
                labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show top words per cluster
            st.subheader("Top Words per Cluster")

            for cluster_id in range(3):
                cluster_comments = df_sample.iloc[clusters == cluster_id]['clean_comments']
                if not cluster_comments.empty:
                    combined_text = ' '.join(cluster_comments)
                    word_counts = Counter(combined_text.split())
                    top_words = pd.DataFrame(word_counts.most_common(10), columns=['Word', 'Count'])

                    st.write(f"### Cluster {cluster_id + 1}")
                    st.dataframe(top_words)
        else:
            st.write("Not enough non-empty comments for clustering visualization.")

        # 5. Attribute Co-occurrence Network
        st.header("Attribute Co-occurrence Network")

        if len(selected_attributes) >= 3:
            # Select the first three attributes
            attrs = selected_attributes[:3]

            # Create nodes (unique values from each attribute)
            nodes = []
            node_colors = []

            for i, attr in enumerate(attrs):
                for val in df[attr].unique():
                    nodes.append(f"{attr}_{val}")
                    node_colors.append(i)  # Color by attribute

            # Create edges (co-occurrences)
            edges = []
            edge_weights = []

            for i, attr1 in enumerate(attrs):
                for j, attr2 in enumerate(attrs):
                    if i < j:  # Avoid duplicates
                        # Count co-occurrences
                        co_occur = pd.crosstab(df[attr1], df[attr2])

                        for val1 in co_occur.index:
                            for val2 in co_occur.columns:
                                weight = co_occur.loc[val1, val2]
                                if weight > 0:
                                    edges.append((f"{attr1}_{val1}", f"{attr2}_{val2}"))
                                    edge_weights.append(weight)

            # Create network graph
            fig = go.Figure()

            # Add edges
            for i, (source, target) in enumerate(edges):
                fig.add_trace(
                    go.Scatter(
                        x=[nodes.index(source), nodes.index(target)],
                        y=[0, 0],
                        mode='lines',
                        line=dict(width=edge_weights[i] / max(edge_weights) * 10, color='gray'),
                        showlegend=False
                    )
                )

            # Add nodes
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(nodes))),
                    y=[0] * len(nodes),
                    mode='markers+text',
                    marker=dict(
                        size=15,
                        color=node_colors,
                        colorscale='Viridis',
                    ),
                    text=nodes,
                    textposition='top center',
                    showlegend=False
                )
            )

            fig.update_layout(
                title='Attribute Co-occurrence Network',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Select at least three attributes to display the co-occurrence network.")

        # 6. Radar Chart for Attribute Distributions
        st.header("Radar Chart for Attribute Distributions")

        if len(selected_attributes) >= 3:
            # Prepare data for radar chart
            categories = selected_attributes

            # Calculate percentage of non-zero values for each attribute
            values = []
            for attr in categories:
                non_zero = (df[attr] != '0').sum() / len(df)
                values.append(non_zero * 100)

            # Create radar chart
            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Attribute Distribution'
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                title='Percentage of Non-Zero Values by Attribute',
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Select at least three attributes to display the radar chart.")



def DataGeneration(data):
    

    # only for testing purpose
    # file_path = 'final_predictions.csv'
    # if True:
    #     final = pd.read_csv('final_predictions.csv')
    #     return final

    
    comments = data['comments']
    cleaned = []
    for text in comments:
        text=str(text)
        cleaned.append(preprocessing(text))
    data['comments'] = cleaned
    # st.dataframe(data)
    data['comments'] = data['comments'].apply(drop_long_sentences)
    data['sarcastic'] = ''
    data['nature']=''
    data['Contextual']=''
    data['ironic']=''
    data['formal']=''
    data['toxic']=''
    json_data = data[['comments','sarcastic','nature','Contextual','ironic','formal','toxic']].to_json(orient='records')
    prompt =f"""
            Your are an expert liguist you have to classifiy the text in field comments into these field
            sarcastic : if the text is sarcastic then fill the field with 1 if non-sarcastic then with 0
            nature : classify text from -2 to 2 where -2 is highly negative and 2 is highly positive and 0 is neutral
            Contextual : classify text into Supportive: 1 , Disgust:2 , Confusion:3 , Optimistic:4 , Pessimistic:5 , Humorous:6 , Mixed_feelings:7
            ironic : if the text is ironic then fill the field with 1 if non-ironic then with 0
            formal : if the text is formal then fill the field with 1 if non-formal then with 0
            toxic : if the text is toxic then fill the field with 1 if non-toxic then with 0
            if text is none then leave the other fields as it is empty
            i have given you text in json file with respective fields please do fill them
        ```
        {json_data}
        ```
    """
    response = model.generate_content(prompt)
    json_data = response.text.replace('`','')
    json_data = json_data.replace('\n','')
    json_data = json_data.replace('json','')
    data_p = json.loads(json_data[:700])
    final = pd.DataFrame(data_p)
    return final






def Enter_Dashboard(data):
    
    if st.button("Return Home"):
        st.session_state.Main = "Home"   
    
    if st.button("Refresh"):
        st.rerun()
    
    df = DataGeneration(data)
    
    graph(df)
    

        
        
    
    


    
    
    
    
    

