import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tqdm import tqdm
import time

print("Starting model training...")

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load and prepare data
print("Loading data...")
df = pd.read_csv('sms-spam.csv')
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df = df.rename(columns={'v1': 'result', 'v2': 'input'})


# Add some modern spam examples
modern_spam = [
    # Tech and Device Scams
    "Click here to win a new iPhone 15! Limited time offer!",
    "Congratulations! You've been selected for a free Samsung Galaxy S24!",
    "Your Apple ID has been locked. Click here to verify: bit.ly/fakelink",
    "Your device has been infected with virus! Click to scan now!",
    
    # Financial Scams
    "CONGRATULATIONS! You've won $5000! Send your bank details to claim",
    "Your bank account will be suspended. Verify here: secure-bank.fake.com",
    "Investment opportunity! 1000% returns guaranteed in crypto trading",
    "Your PayPal account needs verification. Login here: paypal.fake.net",
    "Urgent: Your credit card has been charged $499 for Amazon Prime",
    
    # Shopping Scams
    "Buy now! 90% off on luxury watches www.fakesite.com",
    "AMAZON: Your recent order #45789 has been suspended",
    "Your package delivery failed. Track here: delivery.scam.com",
    "Exclusive offer: Designer bags 80% OFF! Limited stock",
    
    # Subscription Scams
    "FREE Netflix subscription! Click here to claim",
    "Your Netflix payment failed. Update billing: netflix.fake.com",
    "Disney+ subscription: Last chance to claim your free trial",
    
    # Social Media Scams
    "Someone tried to login to your Instagram account. Verify here",
    "Your Facebook account will be disabled. Verify now: fb.fake.com",
    "See who viewed your profile! Click here to check",
    
    # Job Scams
    "Work from home! Earn $5000/week guaranteed",
    "URGENT hiring: $500/day for online data entry jobs",
    "Make money online! No experience needed. $1000/day",
    
    # Contest and Lottery Scams
    "You're our 1,000,000th visitor! Claim your prize now!",
    "Lottery winner alert! You've won £1,000,000! Claim now",
    "CONGRATULATIONS! You've been randomly selected for $10,000",
    
    # Gift Card Scams
    "Get a free $500 Amazon gift card! Click here",
    "Walmart: You've won a $1000 gift voucher. Claim now",
    
    # Cryptocurrency Scams
    "BITCOIN price alert! Invest now for 1000% returns",
    "Exclusive crypto opportunity! Double your investment in 24hrs",
    
    # Dating/Romance Scams
    "Hey beautiful! I saw your profile and wanted to connect ;)",
    "Looking for love? Click here to meet singles in your area!",
    
    # Current Events Scams
    "COVID-19 relief payment pending. Verify details now",
    "Tax refund notification: Claim your pending refund",
    "Government stimulus payment waiting for verification"
]

modern_ham = [
    # Personal Messages
    "Hey, can we meet for coffee tomorrow at 3pm?",
    "Don't forget to bring the documents for the meeting",
    "The dinner was great last night, thanks for cooking!",
    "What time is the team meeting tomorrow?",
    "Could you pick up some groceries on your way home?",
    
    # Work Related
    "Please review the attached report before tomorrow's meeting",
    "The client meeting is rescheduled to 2pm",
    "Can you send me the updated project timeline?",
    
    # Family/Friends
    "Mom, I'll be home late tonight. Don't wait up",
    "Happy birthday! Hope you have a great day",
    "Are we still on for Sunday lunch?",
    
    # Appointments/Reminders
    "Your dental appointment is confirmed for Thursday at 10am",
    "Reminder: Parent-teacher meeting tomorrow at 4pm",
    "Your order #123 has been delivered to your doorstep",
    
    # General Communication
    "The weather is terrible today, drive safely",
    "Thanks for helping me move yesterday",
    "Did you get the notes from today's lecture?",
    "Can you share the recipe for that cake?"
]

# Add modern examples to the dataset
for spam in modern_spam:
    df = df._append({'result': 'spam', 'input': spam}, ignore_index=True)
for ham in modern_ham:
    df = df._append({'result': 'ham', 'input': ham}, ignore_index=True)

# Text preprocessing function
def transform_text(text):
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    # Simple word tokenization using split()
    text = text.split()
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english'):
            y.append(i)
            
    text = y[:]
    y.clear()
    
    ps = PorterStemmer()
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

print("Preprocessing text...")
df['transformed_text'] = df['input'].apply(transform_text)

print("Creating TF-IDF vectors...")
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['result'].apply(lambda x: 1 if x == 'spam' else 0)

print("Training model...")
start_time = time.time()
etc = ExtraTreesClassifier(n_estimators=200, random_state=42)
etc.fit(X, y)
print(f"Training completed in {time.time() - start_time:.2f} seconds")

print("Saving model files...")
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(etc, open('model.pkl', 'wb'))

print("Model files generated successfully!") 