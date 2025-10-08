"""
Logistic Regression Classifier for Disease Outbreak Detection
Enhanced with sentiment analysis and keyword features
Target Accuracy: 91.8%
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, classification_report, confusion_matrix)
from textblob import TextBlob
from scipy.sparse import hstack
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

DISEASE_KEYWORDS = ['flu', 'fever', 'cough', 'sick', 'illness', 'disease', 'virus']

def get_sentiment(text):
    """Extract sentiment polarity"""
    if pd.isna(text) or text == "":
        return 0.0
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return 0.0

def count_keywords(text):
    """Count disease keywords"""
    if pd.isna(text):
        return 0
    return sum(1 for kw in DISEASE_KEYWORDS if kw in text.lower())

def get_text_length(text):
    """Get word count"""
    if pd.isna(text):
        return 0
    return len(text.split())

def train_logistic_regression(data_file='data/processed/tweets_labeled.csv'):
    """
    Train Logistic Regression classifier with enhanced features
    """
    print("="*70)
    print(" "*15 + "LOGISTIC REGRESSION TRAINING")
    print("="*70)
    
    # Load data
    print("\n[1/8] Loading labeled dataset...")
    df = pd.read_csv(data_file)
    print(f"    ✓ Loaded {len(df)} labeled tweets")
    
    # Extract features
    print("\n[2/8] Engineering features...")
    df['sentiment'] = df['clean_text'].apply(get_sentiment)
    df['keyword_count'] = df['clean_text'].apply(count_keywords)
    df['text_length'] = df['clean_text'].apply(get_text_length)
    print("    ✓ Features extracted: sentiment, keyword_count, text_length")
    
    # Prepare data
    X_text = df['clean_text']
    X_sentiment = df['sentiment'].values.reshape(-1, 1)
    X_keywords = df['keyword_count'].values.reshape(-1, 1)
    X_length = df['text_length'].values.reshape(-1, 1)
    y = df['label']
    
    # Split data
    print("\n[3/8] Splitting data...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.3, random_state=42, stratify=y
    )
    X_train_sent, X_test_sent = train_test_split(
        X_sentiment, test_size=0.3, random_state=42, stratify=y
    )
    X_train_kw, X_test_kw = train_test_split(
        X_keywords, test_size=0.3, random_state=42, stratify=y
    )
    X_train_len, X_test_len = train_test_split(
        X_length, test_size=0.3, random_state=42, stratify=y
    )
    
    # TF-IDF
    print("\n[4/8] Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)
    
    # Combine features
    X_train_combined = hstack([X_train_tfidf, X_train_sent, X_train_kw, X_train_len])
    X_test_combined = hstack([X_test_tfidf, X_test_sent, X_test_kw, X_test_len])
    print(f"    ✓ Total features: {X_train_combined.shape[1]}")
    
    # Train model
    print("\n[5/8] Training Logistic Regression model...")
    lr_model = LogisticRegression(
        C=1.0,
        penalty='l2',
        solver='liblinear',
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    lr_model.fit(X_train_combined, y_train)
    print("    ✓ Model trained")
    
    # Predictions
    print("\n[6/8] Making predictions...")
    y_pred = lr_model.predict(X_test_combined)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n" + "="*70)
    print(" "*25 + "MODEL PERFORMANCE")
    print("="*70)
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print("="*70)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                              target_names=['Normal', 'Disease Outbreak']))
    
    # Cross-validation
    print("\n[7/8] Performing 5-fold cross-validation...")
    cv_scores = cross_val_score(lr_model, X_train_combined, y_train, cv=5)
    print(f"    ✓ CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"    ✓ Mean CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")
    
    # Visualizations
    print("\n[8/8] Creating visualizations...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Normal', 'Disease Outbreak'],
                yticklabels=['Normal', 'Disease Outbreak'])
    plt.title(f'Logistic Regression - Confusion Matrix\nAccuracy: {accuracy*100:.2f}%',
              fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/logistic_regression_confusion_matrix.png', dpi=300)
    print("    ✓ Confusion matrix saved")
    
    # Save model
    joblib.dump(lr_model, 'models/logistic_regression_model.pkl')
    joblib.dump(vectorizer, 'models/lr_vectorizer.pkl')
    print("    ✓ Model saved")
    
    # Save results
    results = {
        'Model': 'Logistic Regression',
        'Accuracy': f'{accuracy*100:.2f}%',
        'Precision': f'{precision*100:.2f}%',
        'Recall': f'{recall*100:.2f}%',
        'F1-Score': f'{f1*100:.2f}%'
    }
    pd.DataFrame([results]).to_csv('results/logistic_regression_results.csv', index=False)
    
    print("\n" + "="*70)
    print(" "*15 + "TRAINING COMPLETE!")
    print("="*70)
    
    return lr_model, vectorizer, accuracy

if __name__ == "__main__":
    train_logistic_regression()
