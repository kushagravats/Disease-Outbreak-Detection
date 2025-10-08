"""
Model Comparison Script
Compares Naive Bayes vs Logistic Regression
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compare_models():
    """
    Compare model performances
    """
    print("="*70)
    print(" "*25 + "MODEL COMPARISON")
    print("="*70)
    
    # Load results
    nb_results = pd.read_csv('results/naive_bayes_results.csv')
    lr_results = pd.read_csv('results/logistic_regression_results.csv')
    
    # Extract metrics
    def extract_percentage(value):
        return float(str(value).replace('%', ''))
    
    nb_metrics = {
        'accuracy': extract_percentage(nb_results['Accuracy'].values[0]),
        'precision': extract_percentage(nb_results['Precision'].values[0]),
        'recall': extract_percentage(nb_results['Recall'].values[0]),
        'f1': extract_percentage(nb_results['F1-Score'].values[0])
    }
    
    lr_metrics = {
        'accuracy': extract_percentage(lr_results['Accuracy'].values[0]),
        'precision': extract_percentage(lr_results['Precision'].values[0]),
        'recall': extract_percentage(lr_results['Recall'].values[0]),
        'f1': extract_percentage(lr_results['F1-Score'].values[0])
    }
    
    # Print comparison
    print(f"\n{'Metric':<20} {'Naive Bayes':<20} {'Logistic Regression':<20} {'Difference'}")
    print("="*70)
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        diff = lr_metrics[metric] - nb_metrics[metric]
        print(f"{metric.capitalize():<20} {nb_metrics[metric]:>7.2f}%{'':<12} "
              f"{lr_metrics[metric]:>7.2f}%{'':<12} {diff:>+6.2f}%")
    print("="*70)
    
    # Visualization
    metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    nb_values = [nb_metrics['accuracy'], nb_metrics['precision'],
                 nb_metrics['recall'], nb_metrics['f1']]
    lr_values = [lr_metrics['accuracy'], lr_metrics['precision'],
                 lr_metrics['recall'], lr_metrics['f1']]
    
    x = np.arange(len(metrics_list))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bars1 = ax.bar(x - width/2, nb_values, width, label='Naive Bayes',
                   color='#3498db', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, lr_values, width, label='Logistic Regression',
                   color='#2ecc71', edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Score (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Performance Metrics', fontsize=13, fontweight='bold')
    ax.set_title('Disease Outbreak Detection - Model Performance Comparison\n' +
                 'Early Disease Outbreak Detection through Social Media Mining',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_list, fontsize=12)
    ax.legend(fontsize=12, loc='lower right')
    ax.set_ylim([75, 100])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Comparison chart saved: results/model_comparison.png")
    
    # Save comparison data
    comparison_df = pd.DataFrame({
        'Model': ['Naive Bayes', 'Logistic Regression'],
        'Accuracy (%)': [nb_metrics['accuracy'], lr_metrics['accuracy']],
        'Precision (%)': [nb_metrics['precision'], lr_metrics['precision']],
        'Recall (%)': [nb_metrics['recall'], lr_metrics['recall']],
        'F1-Score (%)': [nb_metrics['f1'], lr_metrics['f1']]
    })
    comparison_df.to_csv('results/model_comparison_results.csv', index=False)
    print("✓ Comparison data saved: results/model_comparison_results.csv")
    
    print("\n" + "="*70)
    print(" "*25 + "COMPARISON COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    compare_models()
