"""
OPTIMIZED Logistic Regression Classifier for Disease Outbreak Detection
Enhanced with advanced features and hyperparameter tuning
Target Accuracy: 90-93% (Higher than Naive Bayes)
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
from scipy.sparse import hstack, csr_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Enhanced disease keywords for outbreak detection
DISEASE_KEYWORDS = [
    'flu', 'fever', 'cough', 'sick', 'illness', 'disease', 'virus',
    'outbreak', 'epidemic', 'infection', 'contagious', 'symptoms',
    'hospital', 'doctor', 'emergency', 'spreading', 'infected'
]

OUTBREAK_KEYWORDS = [
    'outbreak', 'epidemic', 'pandemic', 'spreading', 'cases',
    'confirmed', 'reported', 'health department', 'alert', 'warning'
]

def extract_advanced_features(df):
    """
    Extract advanced features for better classification
    """
    print("    Extracting features...")
    
    # Sentiment analysis
    df['sentiment'] = df['clean_text'].apply(lambda x: 
        TextBlob(str(x)).sentiment.polarity if pd.notna(x) else 0.0
    )

    df['subjectivity'] = df['clean_text'].apply(lambda x:
        TextBlob(str(x)).sentiment.subjectivity if pd.notna(x) else 0.0
    )
    
    df['disease_kw_count'] = df['clean_text'].apply(lambda x:
        sum(1 for kw in DISEASE_KEYWORDS if kw in str(x).lower()) if pd.notna(x) else 0
    )
    
    # Outbreak keyword count
    df['outbreak_kw_count'] = df['clean_text'].apply(lambda x:
        sum(1 for kw in OUTBREAK_KEYWORDS if kw in str(x).lower()) if pd.notna(x) else 0
    )
    
    # Text length features
    df['text_length'] = df['clean_text'].apply(lambda x:
        len(str(x).split()) if pd.notna(x) else 0
    )
    df['char_length'] = df['clean_text'].apply(lambda x:
        len(str(x)) if pd.notna(x) else 0
    )
    
    df['avg_word_length'] = df['clean_text'].apply(lambda x:
        np.mean([len(word) for word in str(x).split()]) if pd.notna(x) and len(str(x).split()) > 0 else 0
    )
    
    df['exclamation_count'] = df['clean_text'].apply(lambda x:
        str(x).count('!') if pd.notna(x) else 0
    )
    df['question_count'] = df['clean_text'].apply(lambda x:
        str(x).count('?') if pd.notna(x) else 0
    )
    
    # Capital letter ratio (shouting/urgency)
    df['capital_ratio'] = df['clean_text'].apply(lambda x:
        sum(1 for c in str(x) if c.isupper()) / len(str(x)) if pd.notna(x) and len(str(x)) > 0 else 0
    )
    
    print(f"    ✓ Extracted {10} advanced features")
    
    return df

def train_logistic_regression(data_file='data/processed/tweets_labeled.csv'):
    """
    Train OPTIMIZED Logistic Regression classifier
    """
    print("="*70)
    print(" "*15 + "LOGISTIC REGRESSION TRAINING")
    print(" "*18 + "(Optimized Version)")
    print("="*70)
    
    # Load data
    print("\n[1/9] Loading labeled dataset...")
    df = pd.read_csv(data_file)
    print(f"    ✓ Loaded {len(df)} labeled tweets")
    
    # Check label distribution
    outbreak_count = (df['label'] == 1).sum()
    normal_count = (df['label'] == 0).sum()
    print(f"\n    Label Distribution:")
    print(f"    - Outbreak: {outbreak_count} ({outbreak_count/len(df)*100:.1f}%)")
    print(f"    - Normal: {normal_count} ({normal_count/len(df)*100:.1f}%)")
    
    print("\n[2/9] Engineering advanced features...")
    df = extract_advanced_features(df)
    
    X_text = df['clean_text']
    feature_cols = ['sentiment', 'subjectivity', 'disease_kw_count', 
                    'outbreak_kw_count', 'text_length', 'char_length',
                    'avg_word_length', 'exclamation_count', 'question_count',
                    'capital_ratio']
    X_features = df[feature_cols].values
    y = df['label'].values
    
    # Split data
    print("\n[3/9] Splitting data (stratified)...")
    X_train_text, X_test_text, X_train_feat, X_test_feat, y_train, y_test = train_test_split(
        X_text, X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    ✓ Training: {len(X_train_text)} samples")
    print(f"    ✓ Testing: {len(X_test_text)} samples")
    
    # TF-IDF Vectorization
    print("\n[4/9] Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=5000,               
        ngram_range=(1, 3),               
        min_df=2,                         
        max_df=0.85,
        sublinear_tf=True,
        strip_accents='unicode',
        lowercase=True,
        token_pattern=r'\b[a-zA-Z]{2,}\b'
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)
    print(f"    ✓ TF-IDF dimensions: {X_train_tfidf.shape[1]}")
    
    # Scale numerical features
    print("\n[5/9] Scaling numerical features...")
    scaler = StandardScaler()
    X_train_feat_scaled = scaler.fit_transform(X_train_feat)
    X_test_feat_scaled = scaler.transform(X_test_feat)
    print(f"    ✓ Scaled {X_train_feat_scaled.shape[1]} numerical features")
    
    # Combine all features
    X_train_combined = hstack([
        X_train_tfidf,
        csr_matrix(X_train_feat_scaled)
    ])
    X_test_combined = hstack([
        X_test_tfidf,
        csr_matrix(X_test_feat_scaled)
    ])
    
    print(f"    ✓ Total features: {X_train_combined.shape[1]}")
    
    # Hyperparameter tuning
    print("\n[6/9] Tuning hyperparameters (this may take a minute)...")
    param_grid = {
        'C': [0.1, 0.5, 1.0, 2.0, 5.0],      # Regularization strength
        'penalty': ['l2'],                     # L2 regularization
        'solver': ['liblinear', 'lbfgs'],      # Optimization algorithms
        'class_weight': ['balanced', None],    # Handle imbalanced data
        'max_iter': [1000]
    }
    
    grid_search = GridSearchCV(
        LogisticRegression(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train_combined, y_train)
    
    print(f"    ✓ Best parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"      • {param}: {value}")
    print(f"    ✓ Best CV score: {grid_search.best_score_*100:.2f}%")
    
    # Use best model
    lr_model = grid_search.best_estimator_
    
    # Train final model
    print("\n[7/9] Training final Logistic Regression model...")
    lr_model.fit(X_train_combined, y_train)
    print("    ✓ Model trained with optimal parameters")
    
    # Predictions
    print("\n[8/9] Making predictions...")
    y_pred = lr_model.predict(X_test_combined)
    y_pred_proba = lr_model.predict_proba(X_test_combined)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print("\n" + "="*70)
    print(" "*25 + "MODEL PERFORMANCE")
    print("="*70)
    print(f"Accuracy:  {accuracy*100:.2f}% 🎯")
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
    print("\n    Performing 5-fold cross-validation...")
    cv_scores = cross_val_score(lr_model, X_train_combined, y_train, cv=5, scoring='accuracy')
    print(f"    ✓ CV Scores: {[f'{s*100:.2f}%' for s in cv_scores]}")
    print(f"    ✓ Mean CV Accuracy: {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)")
    
    # Feature importance analysis
    print("\n Top Predictive Features:")
    
    coefficients = lr_model.coef_[0]
    
    # TF-IDF feature names
    tfidf_features = vectorizer.get_feature_names_out()
    
    top_indices = coefficients.argsort()[-15:][::-1]
    
    print("\n    Top 15 Outbreak Indicators (from TF-IDF):")
    for i, idx in enumerate(top_indices, 1):
        if idx < len(tfidf_features):
            print(f"      {i:2d}. {tfidf_features[idx]}: {coefficients[idx]:.4f}")
    
    # Numerical feature importance
    num_feat_start = len(tfidf_features)
    print("\n    Numerical Feature Importance:")
    for i, feat_name in enumerate(feature_cols):
        feat_idx = num_feat_start + i
        if feat_idx < len(coefficients):
            print(f"      • {feat_name}: {coefficients[feat_idx]:.4f}")
    
    # Visualizations
    print("\n[9/9] Creating visualizations...")
    os.makedirs('results', exist_ok=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Normal', 'Disease Outbreak'],
                yticklabels=['Normal', 'Disease Outbreak'],
                annot_kws={'size': 16})
    plt.title(f'Logistic Regression - Confusion Matrix\nAccuracy: {accuracy*100:.2f}%',
              fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/logistic_regression_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("    ✓ Confusion matrix saved")
    
    # Performance comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Metrics bar chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision, recall, f1]
    colors = ['#27ae60', '#3498db', '#e74c3c', '#f39c12']
    
    bars = ax1.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylim([0, 1])
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Logistic Regression Performance Metrics', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val*100:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Cross-validation scores
    ax2.plot(range(1, 6), cv_scores, marker='o', linewidth=2, markersize=8, color='#27ae60')
    ax2.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean()*100:.2f}%')
    ax2.fill_between(range(1, 6),
                      cv_scores.mean() - cv_scores.std(),
                      cv_scores.mean() + cv_scores.std(),
                      alpha=0.2, color='#27ae60')
    ax2.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('5-Fold Cross-Validation Results', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(1, 6))
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('results/logistic_regression_performance.png', dpi=300, bbox_inches='tight')
    print("    ✓ Performance charts saved")
    
    # Feature importance plot
    fig, ax = plt.subplots(figsize=(12, 8))
    top_n = 20
    top_indices_plot = coefficients.argsort()[-top_n:]
    top_features = [tfidf_features[i] if i < len(tfidf_features) else feature_cols[i - len(tfidf_features)] 
                    for i in top_indices_plot]
    top_coefs = coefficients[top_indices_plot]
    
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_coefs, color='#27ae60', alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Features for Outbreak Detection', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/logistic_regression_feature_importance.png', dpi=300, bbox_inches='tight')
    print("    ✓ Feature importance chart saved")
    
    # Save model, vectorizer, and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(lr_model, 'models/logistic_regression_model.pkl')
    joblib.dump(vectorizer, 'models/lr_vectorizer.pkl')
    joblib.dump(scaler, 'models/lr_scaler.pkl')
    print("    ✓ Model, vectorizer, and scaler saved")
    
    # Save detailed results
    results = {
        'Model': 'Logistic Regression (Optimized)',
        'Accuracy': f'{accuracy*100:.2f}%',
        'Precision': f'{precision*100:.2f}%',
        'Recall': f'{recall*100:.2f}%',
        'F1-Score': f'{f1*100:.2f}%',
        'CV_Mean': f'{cv_scores.mean()*100:.2f}%',
        'CV_Std': f'{cv_scores.std()*100:.2f}%',
        'Best_C': grid_search.best_params_['C'],
        'Best_Solver': grid_search.best_params_['solver'],
        'Class_Weight': grid_search.best_params_['class_weight'],
        'Training_Samples': len(X_train_text),
        'Testing_Samples': len(X_test_text),
        'Total_Features': X_train_combined.shape[1],
        'TF-IDF_Features': X_train_tfidf.shape[1],
        'Numerical_Features': len(feature_cols),
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv('results/logistic_regression_results.csv', index=False)
    
    # Save detailed report
    with open('results/logistic_regression_report.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("LOGISTIC REGRESSION MODEL - TRAINING REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("MODEL CONFIGURATION:\n")
        f.write(f"  Algorithm: Logistic Regression\n")
        f.write(f"  C (regularization): {grid_search.best_params_['C']}\n")
        f.write(f"  Solver: {grid_search.best_params_['solver']}\n")
        f.write(f"  Class Weight: {grid_search.best_params_['class_weight']}\n")
        f.write(f"  Total Features: {X_train_combined.shape[1]}\n")
        f.write(f"    - TF-IDF: {X_train_tfidf.shape[1]}\n")
        f.write(f"    - Numerical: {len(feature_cols)}\n\n")
        
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
        for idx in top_indices[:10]:
            if idx < len(tfidf_features):
                f.write(f"  • {tfidf_features[idx]}: {coefficients[idx]:.4f}\n")
    
    print("    ✓ Detailed report saved")
    
    print("\n" + "="*70)
    print(" "*15 + " TRAINING COMPLETE!")
    print("="*70)
    print(f"\n Files saved:")
    print(f"   • models/logistic_regression_model.pkl")
    print(f"   • models/lr_vectorizer.pkl")
    print(f"   • models/lr_scaler.pkl")
    print(f"   • results/logistic_regression_confusion_matrix.png")
    print(f"   • results/logistic_regression_performance.png")
    print(f"   • results/logistic_regression_feature_importance.png")
    print(f"   • results/logistic_regression_results.csv")
    print(f"   • results/logistic_regression_report.txt")
    
    print(f"\n Expected Performance:")
    print(f"   Logistic Regression should outperform Naive Bayes by 2-5%")
    print(f"   Target: 90-93% accuracy")
    
    return lr_model, vectorizer, scaler, accuracy

if __name__ == "__main__":
    try:
        train_logistic_regression()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()