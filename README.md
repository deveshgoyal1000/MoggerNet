# SMS Spam Detection

## Overview
SMS Spam Detection is a machine learning model that takes an SMS as input and predicts whether the message is a spam or not spam message. The model is built using Python and deployed on the web using Streamlit.

## Technology Used
- Python
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- NLTK
- Matplotlib
- Seaborn
- WordCloud

## Features
- Real-time SMS spam detection
- Multi-language support
- Message statistics
- Word cloud visualization
- Confidence scores
- Spam indicator detection
- Enhanced UI with sidebar options

### Data Collection
The SMS Spam Collection dataset was collected from Kaggle, which contains over 5,500 SMS messages labeled as either spam or not spam.
You can access the dataset from [here](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

### Data Cleaning and Preprocessing
The data was cleaned by handling null and duplicate values, and the "type" column was label-encoded. The data was then preprocessed by converting the text into tokens, removing special characters, stop words and punctuation, and stemming the data. The data was also converted to lowercase before preprocessing.

### Exploratory Data Analysis
Exploratory Data Analysis was performed to gain insights into the dataset. The count of characters, words, and sentences was calculated for each message. The correlation between variables was also calculated, and visualizations were created using pyplots, bar charts, pie charts, 5 number summaries, and heatmaps. Word clouds were also created for spam and non-spam messages, and the most frequent words in spam texts were visualized.

### Model Building and Selection
Multiple classifier models were tried, including NaiveBayes, random forest, KNN, decision tree, logistic regression, ExtraTreesClassifier, and SVC. The best classifier was chosen based on precision, with a precision of 100% achieved.

### Web Deployment
The model was deployed on the web using Streamlit. The user interface has a simple input box where the user can input a message, and the model will predict whether it is spam or not spam.

## Demo
To try out the SMS Spam Detection model, visit [here](https://moggernet-sms-spam-detection.streamlit.app/).

## Usage
To use the SMS Spam Detection model on your own machine, follow these steps:

1. Clone this repository
2. Install the required Python packages:
```bash
pip install -r requirements.txt
```
3. Run the model:
```bash
streamlit run app.py
```
4. Visit localhost:8501 on your web browser

## Contributions
Contributions to this project are welcome. If you find any issues or have suggestions for improvement, please open an issue or a pull request.

## Created By
MoggerNet - 2024


