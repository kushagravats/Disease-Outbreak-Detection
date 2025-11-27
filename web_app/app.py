"""
Enhanced Flask Application for Disease Outbreak Detection
Professional Dashboard with Multiple Pages
"""

from flask import Flask, render_template, request, send_from_directory
import joblib
import re
from textblob import TextBlob
from scipy.sparse import hstack, csr_matrix
import nltk
from nltk.corpus import stopwords
import os
import numpy as np

app = Flask(__name__)

# Load models and vectorizers/scaler
print("Loading models...")
nb_model = joblib.load('../models/naive_bayes_model.pkl')
lr_model = joblib.load('../models/logistic_regression_model.pkl')
nb_vectorizer = joblib.load('../models/nb_vectorizer.pkl')
lr_vectorizer = joblib.load('../models/lr_vectorizer.pkl')
lr_scaler = joblib.load('../models/lr_scaler.pkl')
print("✓ Models loaded successfully!")

DISEASE_KEYWORDS = ['flu', 'fever', 'cough', 'sick', 'illness', 'disease', 'virus',
                   'symptom', 'outbreak', 'epidemic', 'pandemic', 'infection']
OUTBREAK_KEYWORDS = [
    'outbreak', 'epidemic', 'pandemic', 'spreading', 'cases',
    'confirmed', 'reported', 'health department', 'alert', 'warning'
]

def clean_text(text):
    """Preprocess input text"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(words)

def extract_advanced_features_single(text):
    """Extract all 10 numerical features for Logistic Regression prediction"""
    sentiment = TextBlob(text).sentiment.polarity if text else 0.0
    subjectivity = TextBlob(text).sentiment.subjectivity if text else 0.0
    disease_kw_count = sum(1 for kw in DISEASE_KEYWORDS if kw in text)
    outbreak_kw_count = sum(1 for kw in OUTBREAK_KEYWORDS if kw in text)
    text_length = len(text.split())
    char_length = len(text)
    avg_word_length = np.mean([len(word) for word in text.split()]) if len(text.split()) > 0 else 0
    exclamation_count = text.count('!')
    question_count = text.count('?')
    capital_ratio = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0.0

    return np.array([[sentiment, subjectivity, disease_kw_count, outbreak_kw_count,
                      text_length, char_length, avg_word_length,
                      exclamation_count, question_count, capital_ratio]])

@app.route('/')
def home():
    """Dashboard home page"""
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    """Prediction input page"""
    return render_template('predict.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze text and return results"""
    text = request.form['tweet_text']
    clean = clean_text(text)

    text_tfidf_nb = nb_vectorizer.transform([clean])
    nb_pred = nb_model.predict(text_tfidf_nb)[0]
    nb_prob = nb_model.predict_proba(text_tfidf_nb)[0][1] * 100

    text_tfidf_lr = lr_vectorizer.transform([clean])
    features_numeric = extract_advanced_features_single(clean)
    features_numeric_scaled = lr_scaler.transform(features_numeric)
    features_sparse = csr_matrix(features_numeric_scaled)

    features_lr = hstack([text_tfidf_lr, features_sparse])
    lr_pred = lr_model.predict(features_lr)[0]
    lr_prob = lr_model.predict_proba(features_lr)[0][1] * 100

    keyword_count = sum(1 for kw in DISEASE_KEYWORDS if kw in clean)
    word_count = len(clean.split())
    sentiment = features_numeric[0][0]

    result = {
        'text': text,
        'clean_text': clean,
        'nb_result': 'Disease Outbreak Alert' if nb_pred == 1 else 'Normal',
        'lr_result': 'Disease Outbreak Alert' if lr_pred == 1 else 'Normal',
        'nb_confidence': f"{nb_prob:.2f}",
        'lr_confidence': f"{lr_prob:.2f}",
        'nb_class': 'alert' if nb_pred == 1 else 'safe',
        'lr_class': 'alert' if lr_pred == 1 else 'safe',
        'consensus': 'OUTBREAK DETECTED' if (nb_pred == 1 and lr_pred == 1) else
                     'POSSIBLE OUTBREAK' if (nb_pred == 1 or lr_pred == 1) else
                     'NO OUTBREAK DETECTED',
        'sentiment': f"{sentiment:.3f}",
        'keyword_count': keyword_count,
        'word_count': word_count
    }

    return render_template('result.html', result=result)

@app.route('/analytics')
def analytics():
    """Model analytics and comparison page"""
    return render_template('analytics.html')

@app.route('/about')
def about():
    """About the project page"""
    return render_template('about.html')

@app.route('/results/<path:filename>')
def serve_results(filename):
    """Serve result images"""
    return send_from_directory('../results', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
