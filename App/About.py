import streamlit as st

def about_us():
    st.title("About Us")
    st.write("Welcome to our application!")
    st.write("At SentimentSense, we are dedicated to revolutionizing sentiment analysis by integrating advanced sarcasm detection and contextual understanding into our framework. Our mission is to provide businesses, researchers, and social media analysts with accurate, real-time insights from online conversations, customer feedback, and social media interactions. Unlike traditional sentiment analysis tools, SentimentSense goes beyond basic polarity detection by interpreting sarcasm, slang, and conversational context to deliver deeper, more meaningful insights. With multilingual support, interactive visualizations, and scalable APIs, our solution helps brands, content creators, and organizations make data-driven decisions with confidence. Join us in redefining sentiment analysis with intelligence and precision!")
    st.write("Created  By :- Anubhav Singh, Nishant Kalane, Sahil Nagpure, Shubhankar Warkade")
    # Back to Home Button
    if st.button("Back to Home"):
        st.session_state["page"] = "Home"
