# MoggerNet: SMS Spam Detection with Enterprise Infrastructure

## Project Overview
A production-grade SMS spam detection system with high availability and disaster recovery capabilities. The system combines machine learning for spam detection with enterprise-level DevOps practices.

## Core Components

### 1. SMS Spam Detection System
- Real-time SMS spam detection using machine learning
- Multi-language support with 100% precision
- Interactive UI with word cloud visualization
- Built with Python, Scikit-learn, NLTK, and Streamlit

### Data Collection
The SMS Spam Collection dataset was collected from Kaggle, which contains over 5,500 SMS messages labeled as either spam or not spam.
You can access the dataset from [here](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

### Data Cleaning and Preprocessing
The data was cleaned by handling null and duplicate values, and the "type" column was label-encoded. The data 
was then preprocessed by converting the text into tokens, removing special characters, stop words and 
punctuation, and stemming the data. The data was also converted to lowercase before preprocessing.
### Exploratory Data Analysis

### Enterprise Infrastructure
- Fault-tolerant Jenkins infrastructure with 2-node HA cluster using Kubernetes StatefulSets
- Automated multi-region backup strategy with 24-hour RPO using AWS S3
- Automated disaster recovery system with <5 minute RTO using AWS Route53
- Comprehensive monitoring using Prometheus/Grafana with Slack alerts
- RBAC-based access control system

## Technology Stack
- **ML/Data Science**: Python, Scikit-learn, Pandas, NumPy, NLTK
- **Web Framework**: Streamlit
- **DevOps Tools**: Jenkins, Kubernetes, Docker
- **Monitoring**: Prometheus, Grafana, AlertManager
- **Cloud Services**: AWS (S3, Route53)
- **Visualization**: Matplotlib, Seaborn, WordCloud

## Features
- Real-time spam detection
- Automated CI/CD pipelines
- Cross-region disaster recovery
- Real-time system monitoring
- Secure access control
- Automated backup system

### Web Deployment
The model was deployed on the web using Streamlit. The user interface has a simple input box where the user can input a message, and the model will predict whether it is spam or not spam.

## Demo
To try out the SMS Spam Detection model, visit [here](https://moggernet-sms-spam-detection.streamlit.app/).

## Local Development Setup
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Start the application:
```bash
streamlit run app.py
```
4. Deploy infrastructure:
```bash
kubectl apply -f k8s-manifests/
```

## Infrastructure Deployment
1. Configure Kubernetes cluster
2. Apply Kubernetes manifests
3. Setup monitoring stack
4. Configure backup system
5. Verify HA setup

## Security
- Role-based access control (RBAC)
- Secure credential management
- Automated backup system
- Real-time security monitoring

## Monitoring & Alerts
- System metrics visualization
- Real-time performance monitoring
- Automated Slack notifications
- Custom alert thresholds

## Contributions
Contributions are welcome! Please feel free to submit pull requests.

## Created By
MoggerNet - 2024


