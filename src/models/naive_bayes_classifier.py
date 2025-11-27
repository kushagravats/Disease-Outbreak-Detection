"""
OPTIMIZED Naive Bayes Classifier for Disease Outbreak Detection
Enhanced with better preprocessing and feature selection
Target Accuracy: 86-89%
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, classification_report, confusion_matrix)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def train_naive_bayes(data_file='data/processed/tweets_labeled.csv'):
    """
    Train OPTIMIZED Naive Bayes classifier
    """
    print("="*70)
    print(" "*20 + "NAIVE BAYES TRAINING")
    print(" "*23 + "(Optimized Version)")
    print("="*70)
    
    # Load data
    print("\n[1/8] Loading labeled dataset...")
    df = pd.read_csv(data_file)
    print(f"    ✓ Loaded {len(df)} labeled tweets")
    
    # Check label distribution
    outbreak_count = (df['label'] == 1).sum()
    normal_count = (df['label'] == 0).sum()
    print(f"\n    Label Distribution:")
    print(f"    - Outbreak: {outbreak_count} ({outbreak_count/len(df)*100:.1f}%)")
    print(f"    - Normal: {normal_count} ({normal_count/len(df)*100:.1f}%)")
    
    X = df['clean_text']
    y = df['label']
    
    # Split data
    print("\n[2/8] Splitting data (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    ✓ Training: {len(X_train)} samples")
    print(f"    ✓ Testing: {len(X_test)} samples")
    
    # TF-IDF Vectorization with optimized parameters
    print("\n[3/8] Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=3000,           # Reduced for NB
        ngram_range=(1, 2),           # Unigrams and bigrams
        min_df=3,                     # Ignore rare terms
        max_df=0.85,                  # Ignore too common terms
        sublinear_tf=True,            # Use log scaling
        strip_accents='unicode',
        lowercase=True,
        token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only words with 2+ chars
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"    ✓ Feature dimensions: {X_train_tfidf.shape[1]}")
    print(f"    ✓ Sparsity: {(1.0 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1]))*100:.2f}%")
    
    # Hyperparameter tuning with GridSearch
    print("\n[4/8] Tuning hyperparameters...")
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],  # Smoothing parameter
        'fit_prior': [True, False]
    }
    
    grid_search = GridSearchCV(
        MultinomialNB(),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train_tfidf, y_train)
    
    print(f"    ✓ Best parameters: {grid_search.best_params_}")
    print(f"    ✓ Best CV score: {grid_search.best_score_*100:.2f}%")
    
    # Use best model
    nb_model = grid_search.best_estimator_
    
    # Train final model
    print("\n[5/8] Training final Naive Bayes model...")
    nb_model.fit(X_train_tfidf, y_train)
    print("    ✓ Model trained with optimal parameters")
    
    # Predictions
    print("\n[6/8] Making predictions...")
    y_pred = nb_model.predict(X_test_tfidf)
    y_pred_proba = nb_model.predict_proba(X_test_tfidf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print("\n" + "="*70)
    print(" "*25 + "MODEL PERFORMANCE")
    print("="*70)
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print("="*70)
    
    print("\n Detailed Classification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Normal', 'Disease Outbreak'],
                              digits=4))
    
    # Class-specific metrics
    print("\n Class-Specific Performance:")
    for i, label in enumerate(['Normal', 'Disease Outbreak']):
        mask = y_test == i
        if mask.sum() > 0:
            class_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"    {label}: {class_acc*100:.2f}% accuracy on {mask.sum()} samples")
    
    # Cross-validation
    print("\n[7/8] Performing 5-fold cross-validation...")
    cv_scores = cross_val_score(nb_model, X_train_tfidf, y_train, cv=5, scoring='accuracy')
    print(f"    ✓ CV Scores: {[f'{s*100:.2f}%' for s in cv_scores]}")
    print(f"    ✓ Mean CV Accuracy: {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)")
    
    # Feature importance analysis
    print("\n🔍 Top Predictive Features:")
    feature_names = vectorizer.get_feature_names_out()
    
    # Features most indicative of outbreak
    outbreak_log_prob = nb_model.feature_log_prob_[1]  # Class 1 (outbreak)
    normal_log_prob = nb_model.feature_log_prob_[0]    # Class 0 (normal)
    
    # Calculate feature importance (difference in log probabilities)
    feature_importance = outbreak_log_prob - normal_log_prob
    top_outbreak_indices = feature_importance.argsort()[-10:][::-1]
    
    print("\n    Top 10 Outbreak Indicators:")
    for idx in top_outbreak_indices:
        print(f"      • {feature_names[idx]}: {feature_importance[idx]:.4f}")
    
    # Visualizations
    print("\n[8/8] Creating visualizations...")
    os.makedirs('results', exist_ok=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Disease Outbreak'],
                yticklabels=['Normal', 'Disease Outbreak'],
                annot_kws={'size': 16})
    plt.title(f'Naive Bayes - Confusion Matrix\nAccuracy: {accuracy*100:.2f}%',
              fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/naive_bayes_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("    ✓ Confusion matrix saved")
    
    # Performance comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Metrics bar chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision, recall, f1]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    bars = ax1.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylim([0, 1])
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Naive Bayes Performance Metrics', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val*100:.1f}%', ha='center', va='bottom', fontweight='bold')
    
   
    ax2.plot(range(1, 6), cv_scores, marker='o', linewidth=2, markersize=8, color='#3498db')
    ax2.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean()*100:.2f}%')
    ax2.fill_between(range(1, 6), 
                      cv_scores.mean() - cv_scores.std(), 
                      cv_scores.mean() + cv_scores.std(), 
                      alpha=0.2, color='#3498db')
    ax2.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('5-Fold Cross-Validation Results', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(1, 6))
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('results/naive_bayes_performance.png', dpi=300, bbox_inches='tight')
    print("    ✓ Performance charts saved")
    
    # Save model and vectorizer
    os.makedirs('models', exist_ok=True)
    joblib.dump(nb_model, 'models/naive_bayes_model.pkl')
    joblib.dump(vectorizer, 'models/nb_vectorizer.pkl')
    print("    ✓ Model and vectorizer saved")
    
    # Save detailed results
    results = {
        'Model': 'Naive Bayes (Optimized)',
        'Accuracy': f'{accuracy*100:.2f}%',
        'Precision': f'{precision*100:.2f}%',
        'Recall': f'{recall*100:.2f}%',
        'F1-Score': f'{f1*100:.2f}%',
        'CV_Mean': f'{cv_scores.mean()*100:.2f}%',
        'CV_Std': f'{cv_scores.std()*100:.2f}%',
        'Best_Alpha': grid_search.best_params_['alpha'],
        'Best_Fit_Prior': grid_search.best_params_['fit_prior'],
        'Training_Samples': len(X_train),
        'Testing_Samples': len(X_test),
        'Features': X_train_tfidf.shape[1],
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv('results/naive_bayes_results.csv', index=False)
    
    # Save detailed report
    with open('results/naive_bayes_report.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("NAIVE BAYES MODEL - TRAINING REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("MODEL CONFIGURATION:\n")
        f.write(f"  Algorithm: Multinomial Naive Bayes\n")
        f.write(f"  Alpha (smoothing): {grid_search.best_params_['alpha']}\n")
        f.write(f"  Fit Prior: {grid_search.best_params_['fit_prior']}\n")
        f.write(f"  Features: {X_train_tfidf.shape[1]}\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write(f"  Accuracy:  {accuracy*100:.2f}%\n")
        f.write(f"  Precision: {precision*100:.2f}%\n")
        f.write(f"  Recall:    {recall*100:.2f}%\n")
        f.write(f"  F1-Score:  {f1*100:.2f}%\n\n")
        
        f.write("CROSS-VALIDATION:\n")
        f.write(f"  Mean: {cv_scores.mean()*100:.2f}%\n")
        f.write(f"  Std:  {cv_scores.std()*100:.2f}%\n")
        f.write(f"  Scores: {[f'{s*100:.2f}%' for s in cv_scores]}\n\n")
        
        f.write("CONFUSION MATRIX:\n")
        f.write(f"  {cm}\n\n")
        
        f.write("TOP OUTBREAK INDICATORS:\n")
        for idx in top_outbreak_indices:
            f.write(f"  • {feature_names[idx]}\n")
    
    print("    ✓ Detailed report saved")
    
    print("\n" + "="*70)
    print(" "*20 + "TRAINING COMPLETE!")
    print("="*70)
    print(f"\n Files saved:")
    print(f"   • models/naive_bayes_model.pkl")
    print(f"   • models/nb_vectorizer.pkl")
    print(f"   • results/naive_bayes_confusion_matrix.png")
    print(f"   • results/naive_bayes_performance.png")
    print(f"   • results/naive_bayes_results.csv")
    print(f"   • results/naive_bayes_report.txt")
    
    return nb_model, vectorizer, accuracy

if __name__ == "__main__":
    try:
        train_naive_bayes()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()