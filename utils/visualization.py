import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import streamlit as st

def plot_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

def plot_confidence(spam_prob):
    fig, ax = plt.subplots(figsize=(10, 2))
    sns.barplot(x=[spam_prob, 1-spam_prob], y=['Spam', 'Not Spam'])
    plt.title('Prediction Confidence')
    st.pyplot(fig)

def show_message_stats(text):
    st.write("Message Statistics:")
    stats = {
        "Total Characters": len(text),
        "Word Count": len(text.split()),
        "Special Characters": sum(not c.isalnum() for c in text),
        "Numbers": sum(c.isdigit() for c in text)
    }
    return stats 