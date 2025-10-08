"""
Naive Bayes Classifier for Disease Outbreak Detection
Model Specifications: MultinomialNB with alpha=1.0, TF-IDF features
Target Accuracy: 88.5%
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, classification_report, confusion_matrix)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_naive_bayes(data_file='data/processed/tweets_labeled.csv'):
    """
    Train Naive Bayes classifier
    """
    print("="*70)
    print(" "*20 + "NAIVE BAYES TRAINING")
    print("="*70)
    
    # Load data
    print("\n[1/7] Loading labeled dataset...")
    df = pd.read_csv(data_file)
    print(f"    ✓ Loaded {len(df)} labeled tweets")
    
    X = df['clean_text']
    y = df['label']
    
    print("\n[2/7] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"    ✓ Training samples: {len(X_train)}")
    print(f"    ✓ Testing samples: {len(X_test)}")
    
    # TF-IDF Vectorization
    print("\n[3/7] Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"    ✓ Feature dimensions: {X_train_tfidf.shape[1]}")
    
    # Train model
    print("\n[4/7] Training Naive Bayes model...")
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(X_train_tfidf, y_train)
    print("    ✓ Model trained")
    
    # Predictions
    print("\n[5/7] Making predictions...")
    y_pred = nb_model.predict(X_test_tfidf)
    
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
    print("\n[6/7] Performing 5-fold cross-validation...")
    cv_scores = cross_val_score(nb_model, X_train_tfidf, y_train, cv=5)
    print(f"    ✓ CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"    ✓ Mean CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")
    
    # Visualizations
    print("\n[7/7] Creating visualizations...")
    os.makedirs('results', exist_ok=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Disease Outbreak'],
                yticklabels=['Normal', 'Disease Outbreak'])
    plt.title(f'Naive Bayes - Confusion Matrix\nAccuracy: {accuracy*100:.2f}%',
              fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/naive_bayes_confusion_matrix.png', dpi=300)
    print("    ✓ Confusion matrix saved")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(nb_model, 'models/naive_bayes_model.pkl')
    joblib.dump(vectorizer, 'models/nb_vectorizer.pkl')
    print("    ✓ Model saved")
    
    # Save results
    results = {
        'Model': 'Naive Bayes',
        'Accuracy': f'{accuracy*100:.2f}%',
        'Precision': f'{precision*100:.2f}%',
        'Recall': f'{recall*100:.2f}%',
        'F1-Score': f'{f1*100:.2f}%'
    }
    pd.DataFrame([results]).to_csv('results/naive_bayes_results.csv', index=False)
    
    print("\n" + "="*70)
    print(" "*20 + "TRAINING COMPLETE!")
    print("="*70)
    
    return nb_model, vectorizer, accuracy

if __name__ == "__main__":
    train_naive_bayes()
