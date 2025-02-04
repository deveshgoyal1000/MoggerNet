import nltk
import os

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

import streamlit as st
import pickle 
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from utils.preprocessing import transform_text, translate_text
from utils.visualization import plot_wordcloud, plot_confidence, show_message_stats

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tk = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

# Page config
st.set_page_config(page_title="SMS Spam Detection", layout="wide")

# Sidebar
st.sidebar.title("Options")
show_stats = st.sidebar.checkbox("Show Message Statistics")
show_wordcloud = st.sidebar.checkbox("Show Word Cloud")
show_confidence = st.sidebar.checkbox("Show Confidence Score")
language = st.sidebar.selectbox("Select Language", ["English", "Spanish", "French", "German"])

# Main content
st.title("SMS Spam Detection Model")
st.write("*MoggerNet*")
    

input_sms = st.text_area("Enter the SMS")

if st.button('Predict'):
    if input_sms:
        # Translate if needed
        if language != "English":
            input_sms = translate_text(input_sms, 'en')
            st.info(f"Translated text: {input_sms}")

        # Preprocess
        transformed_sms = transform_text(input_sms)
        
        # Vectorize
        vector_input = tk.transform([transformed_sms])
        
        # Predict
        result = model.predict(vector_input)[0]
        proba = model.predict_proba(vector_input)[0]

        # Show result
        if result == 1:
            st.error("Spam")
        else:
            st.success("Not Spam")

        # Show additional information
        if show_confidence:
            plot_confidence(proba[1])

        if show_stats:
            stats = show_message_stats(input_sms)
            st.write(stats)

        if show_wordcloud:
            plot_wordcloud(input_sms)

        # Spam indicators
        spam_words = ['free', 'win', 'cash', 'prize', 'click', 'urgent']
        found_indicators = [word for word in spam_words if word in input_sms.lower()]
        if found_indicators:
            st.warning(f"Spam indicators found: {', '.join(found_indicators)}")
