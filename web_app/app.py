"""
Enhanced Flask Application for Disease Outbreak Detection
Professional Dashboard with Multiple Pages
"""

from flask import Flask, render_template, request, send_from_directory
import joblib
import re
from textblob import TextBlob
from scipy.sparse import hstack
import nltk
from nltk.corpus import stopwords
import os

app = Flask(__name__)

# Load models
print("Loading models...")
nb_model = joblib.load('../models/naive_bayes_model.pkl')
lr_model = joblib.load('../models/logistic_regression_model.pkl')
nb_vectorizer = joblib.load('../models/nb_vectorizer.pkl')
lr_vectorizer = joblib.load('../models/lr_vectorizer.pkl')
print("✓ Models loaded successfully!")

DISEASE_KEYWORDS = ['flu', 'fever', 'cough', 'sick', 'illness', 'disease', 'virus',
                   'symptom', 'outbreak', 'epidemic', 'pandemic', 'infection']

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
    
    # Naive Bayes prediction
    text_tfidf_nb = nb_vectorizer.transform([clean])
    nb_pred = nb_model.predict(text_tfidf_nb)[0]
    nb_prob = nb_model.predict_proba(text_tfidf_nb)[0][1] * 100
    
    # Logistic Regression prediction
    text_tfidf_lr = lr_vectorizer.transform([clean])
    sentiment = TextBlob(clean).sentiment.polarity if clean else 0.0
    keyword_count = sum(1 for kw in DISEASE_KEYWORDS if kw in clean.lower())
    text_length = len(clean.split())
    
    features_lr = hstack([text_tfidf_lr, [[sentiment]], [[keyword_count]], [[text_length]]])
    lr_pred = lr_model.predict(features_lr)[0]
    lr_prob = lr_model.predict_proba(features_lr)[0][1] * 100
    
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
        'word_count': text_length
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
