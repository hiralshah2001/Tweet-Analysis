import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
import plotly.express as px
import pickle
import re

# Load the test dataset
data = pd.read_csv("C:/Users/Dell/OneDrive/Desktop/sem 3/NLP/ass-4/Sentiment_Data.csv", encoding='ISO-8859-1', nrows=10000)

# Preprocess tweets
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+', '', text)
    text = re.sub(r'[^A-Za-z0-9\s#]', '', text)
    text = text.lower()
    return text

data['processed_text'] = data['Tweet'].apply(clean_text)

# Load the trained LSTM model and tokenizer
model = load_model('C:/Users/Dell/OneDrive/Desktop/sem 3/deploymnet/Final Presentation/lstm_model.h5')
with open('C:/Users/Dell/OneDrive/Desktop/sem 3/deploymnet/Final Presentation/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Sentiment Analysis
def analyze_sentiment(filtered_tweets, tokenizer, model):
    if filtered_tweets.empty:
        return {}, "No tweets found for this hashtag."
    
    sequences = tokenizer.texts_to_sequences(filtered_tweets['processed_text'])
    padded_sequences = pad_sequences(sequences, maxlen=100, padding='post')
    predictions = model.predict(padded_sequences)
    sentiment_classes = np.argmax(predictions, axis=1)

    sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    risk_mapping = {"Negative": "High Risk", "Neutral": "Medium Risk", "Positive": "Low Risk"}
    sentiments = pd.Series(sentiment_classes).map(sentiment_mapping)
    risks = sentiments.map(risk_mapping).value_counts(normalize=True) * 100
    dominant_risk = risks.idxmax()
    return risks.to_dict(), dominant_risk

# Generate Hashtag Word Cloud Data
def generate_hashtag_wordcloud_data(hashtags):
    word_freq = Counter(hashtags)
    hashtag_df = pd.DataFrame(word_freq.most_common(20), columns=["Hashtag", "Frequency"])
    return hashtag_df

# Streamlit App Configuration
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide", page_icon="📊")

# Advanced Styling
st.markdown("""
    <style>
    body {
        background: linear-gradient(180deg, #e0f7fa, #f3f8ff);
        font-family: 'Arial', sans-serif;
    }
    .header {
        text-align: center; 
        font-size: 48px; 
        color: #3b5998; 
        margin-bottom: 10px;
    }
    .sub-header {
        text-align: center; 
        font-size: 24px; 
        color: #4caf50;
    }
    .content-container {
        margin: 20px auto; 
        padding: 20px; 
        background-color: #ffffff; 
        border-radius: 10px; 
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        max-width: 800px;
    }
    .center {
        text-align: center;
    }
    .hashtag-box {
        display: flex;
        justify-content: center;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 10px;
    }
    .hashtag {
        background-color: #e3f2fd;
        padding: 8px 12px;
        border-radius: 5px;
        font-weight: bold;
        color: #007bff;
        text-decoration: none;
        cursor: pointer;
    }
    .tweet-card {
        margin: 10px auto; 
        padding: 15px; 
        background-color: #f9f9f9; 
        border-radius: 10px; 
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);
        width: 80%;  
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class='header'>
        <img src='https://freepnglogo.com/images/all_img/1707222563twitter-logo-png.png' alt='Logo' style='display: block; margin: 0 auto; width: 100px; height: auto;'>
        <h1 style='text-align: center; font-size: 48px; color: #3b5998;'>Sentiment Analysis Dashboard</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown("<div class='sub-header'>Discover Market trends and risks through tweets</div>", unsafe_allow_html=True)

# Input Section
st.markdown("<div class='content-container'>", unsafe_allow_html=True)
hashtag = st.text_input("Enter a hashtag (e.g., #Tesla):")

if hashtag:
    filtered_tweets = data[data['Tweet'].str.contains(hashtag, case=False, na=False)]
    if filtered_tweets.empty:
        st.error("No tweets found for this hashtag.")
    else:
        sentiment_summary, market_trend = analyze_sentiment(filtered_tweets, tokenizer, model)

        # Market Trend Display
        st.markdown(f"<h3 class='center'>Market Trend: <span style='color:#4caf50;'>{market_trend}</span></h3>", unsafe_allow_html=True)

        # Interactive Pie Chart
        pie_data = pd.DataFrame(list(sentiment_summary.items()), columns=["Sentiment", "Percentage"])
        fig = px.pie(pie_data, values="Percentage", names="Sentiment", title="Risk Classification", 
                     color_discrete_sequence=["#4CAF50", "#FFC107", "#F44336"])
        st.plotly_chart(fig, use_container_width=True)

        # Hashtag Word Cloud Bar Chart
        st.markdown("<h4 class='center'>Hashtag Word Cloud</h4>", unsafe_allow_html=True)
        hashtags = Counter(filtered_tweets['Tweet'].str.findall(r"#\w+").sum())
        hashtag_data = generate_hashtag_wordcloud_data(hashtags)
        wordcloud_fig = px.bar(hashtag_data, x="Hashtag", y="Frequency", title="Hashtag Frequencies", 
                               color="Frequency", color_continuous_scale="Viridis")
        wordcloud_fig.update_layout(xaxis_title="Hashtag", yaxis_title="Frequency")
        st.plotly_chart(wordcloud_fig, use_container_width=True)

        # Centered Trending Hashtags
        st.markdown("<div class='center'>", unsafe_allow_html=True)
        st.markdown("<h4>Trending Hashtags</h4>", unsafe_allow_html=True)
        st.markdown("<div class='hashtag-box'>", unsafe_allow_html=True)
        for tag, count in hashtags.most_common(5):
            st.markdown(f"<a href='https://twitter.com/hashtag/{tag}' target='_blank' class='hashtag'>{tag}</a>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Recent Tweets
        st.markdown("<h4 class='center'>Recent Tweets</h4>", unsafe_allow_html=True)
        for tweet in filtered_tweets['Tweet'].head(10):
            highlighted_tweet = re.sub(r"(https?://\S+)", r"<a href='\1' target='_blank'>\1</a>", tweet)
            st.markdown(f"<div class='tweet-card'>{highlighted_tweet}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
