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
from utils.visualization import plot_wordcloud, plot_confidence, show_message_stats, show_performance_metrics, plot_confusion_matrix, plot_roc_curve, show_detailed_metrics
import base64

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


# Page configuration
st.set_page_config(
    page_title="SMS Spam Detector | MoggerNet",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        height: 3rem;
        font-size: 1.2rem;
        margin-top: 1rem;
    }
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #262730;
    }
    .st-emotion-cache-1v0mbdj {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return tfidf, model

@st.cache_resource
def load_test_data():
    try:
        test_data = pickle.load(open('test_data.pkl', 'rb'))
        return test_data
    except Exception as e:
        st.error(f"Error loading test data: {str(e)}")
        return None

tfidf, model = load_models()

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")
    
    language = st.selectbox(
        "🌐 Select Language",
        ["English", "Spanish", "French", "German", "Hindi"]
    )
    
    st.markdown("### 📊 Analysis Options")
    show_confidence = st.checkbox("Show Confidence Score", value=True)
    show_stats = st.checkbox("Show Message Statistics", value=True)
    show_wordcloud = st.checkbox("Show Word Cloud")
    
    st.markdown("---")
    st.markdown("### 🛠️ About")
    st.info("""
        This SMS Spam Detector uses machine learning to identify spam messages.
        
        Built with ❤️ by MoggerNet
    """)

# Main content
col1, col2, col3 = st.columns([1,6,1])
with col2:
    st.title("🛡️ SMS Spam Detector")
    st.markdown("### Protect yourself from unwanted messages")
    
    # After the title
    tab1, tab2 = st.tabs(["Message Analysis", "Model Performance"])

    with tab1:
        # Move all the existing message input and analysis code here
        input_sms = st.text_area(
            "Enter your message here",
            value="",
            height=150,
            placeholder="Type or paste your message here..."
        )
        
        col_button1, col_button2 = st.columns(2)
        with col_button1:
            predict_button = st.button("Analyze Message")
        with col_button2:
            # Simplified clear button
            if st.button("🗑️ Clear"):
                st.rerun()

        if predict_button and input_sms:
            st.markdown("---")
            
            # Progress bar for analysis
            with st.spinner("Analyzing message..."):
                # Translate if needed
                if language != "English":
                    input_sms = translate_text(input_sms, 'en')
                    st.info(f"🔄 Translated text: {input_sms}")
                
                # Preprocess and predict
                transformed_sms = transform_text(input_sms)
                vector_input = tfidf.transform([transformed_sms])
                result = model.predict(vector_input)[0]
                proba = model.predict_proba(vector_input)[0]
                
                # Display results
                col_result1, col_result2 = st.columns(2)
                
                with col_result1:
                    if result == 1:
                        st.error("🚨 Spam Detected!")
                        st.markdown(f"Confidence: **{proba[1]*100:.2f}%**")
                    else:
                        st.success("✅ Not Spam")
                        st.markdown(f"Confidence: **{proba[0]*100:.2f}%**")
                
                with col_result2:
                    if show_confidence:
                        plot_confidence(proba[1])
                
                # Additional analysis
                if show_stats or show_wordcloud:
                    st.markdown("### 📊 Detailed Analysis")
                    
                    col_analysis1, col_analysis2 = st.columns(2)
                    
                    with col_analysis1:
                        if show_stats:
                            stats = show_message_stats(input_sms)
                            st.write("#### Message Statistics")
                            for key, value in stats.items():
                                st.markdown(f"**{key}:** {value}")
                    
                    with col_analysis2:
                        if show_wordcloud:
                            st.write("#### Word Cloud")
                            plot_wordcloud(input_sms)
                
                # Spam indicators
                spam_words = ['free', 'win', 'cash', 'prize', 'click', 'urgent', 'limited', 'offer', 'congratulations', 'credit card', 'bitcoin']
                found_indicators = [word for word in spam_words if word in input_sms.lower()]
                if found_indicators:
                    st.warning(f"⚠️ Potential spam indicators found: {', '.join(found_indicators)}")

    with tab2:
        st.markdown("### 📊 Model Performance")
        test_data = load_test_data()
        if test_data is None:
            st.error("Could not load test data. Please run train_model.py first.")
        else:
            try:
                # Basic Metrics
                metrics = show_performance_metrics(test_data['y_test'], test_data['y_pred'])
                
                st.markdown("#### 📈 Basic Performance Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{metrics['Accuracy']*100:.2f}%")
                with col2:
                    st.metric("Precision", f"{metrics['Precision']*100:.2f}%")
                with col3:
                    st.metric("Recall", f"{metrics['Recall']*100:.2f}%")
                with col4:
                    st.metric("F1 Score", f"{metrics['F1 Score']*100:.2f}%")
                
                # ... rest of the metrics code ...
            except Exception as e:
                st.error(f"Error calculating metrics: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>MoggerNet, Deveshgoyal1000</p>
    </div>
    """,
    unsafe_allow_html=True
)
