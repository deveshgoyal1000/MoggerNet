import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

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

def show_performance_metrics(y_true, y_pred):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred)
    }
    return metrics

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    st.pyplot(plt) 