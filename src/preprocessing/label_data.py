"""
Data Labeling Module
Labels tweets as disease-related or normal using keyword matching
with advanced validation and quality metrics

Features:
- Multi-keyword matching with configurable threshold
- Quality metrics and validation
- Balanced dataset generation
- Detailed logging and statistics
- Sample verification
"""

import pandas as pd
import os
import sys
from datetime import datetime
from collections import Counter

# Disease-related keywords (comprehensive list)
DISEASE_KEYWORDS = [
    'flu', 'fever', 'cough', 'sick', 'illness', 'disease', 'virus',
    'symptom', 'outbreak', 'epidemic', 'pandemic', 'infection',
    'covid', 'corona', 'coronavirus', 'headache', 'pain', 
    'hospital', 'emergency', 'breathing', 'pneumonia', 'vomiting',
    'diarrhea', 'fatigue', 'spread', 'contagious', 'diagnosed',
    'tested', 'positive', 'patient', 'clinic', 'doctor', 'ache',
    'sore', 'throat', 'runny', 'nose', 'chills', 'nausea',
    'quarantine', 'isolate', 'vaccine', 'treatment'
]

# Configuration
KEYWORD_THRESHOLD = 2  # Minimum keywords required for disease label
LOG_FILE = 'logs/labeling.log'

def setup_logging():
    """Initialize logging system"""
    os.makedirs('logs', exist_ok=True)
    return open(LOG_FILE, 'a', encoding='utf-8')

def log_message(log_file, message, print_message=True):
    """Write message to both console and log file"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    if print_message:
        print(message)
    log_file.write(log_entry + '\n')
    log_file.flush()

def print_header(title):
    """Print formatted section header"""
    print("\n" + "="*60)
    print(" "*((60 - len(title))//2) + title)
    print("="*60)

def find_keywords_in_text(text):
    """
    Find which keywords appear in text
    Returns: list of matched keywords
    """
    if pd.isna(text) or text == "":
        return []
    
    text_lower = text.lower()
    matched_keywords = [kw for kw in DISEASE_KEYWORDS if kw in text_lower]
    return matched_keywords

def label_tweet(text):
    """
    Label tweet as disease-related (1) or normal (0)
    Requires at least KEYWORD_THRESHOLD keywords to reduce false positives
    """
    matched_keywords = find_keywords_in_text(text)
    keyword_count = len(matched_keywords)
    return 1 if keyword_count >= KEYWORD_THRESHOLD else 0

def calculate_confidence(text):
    """
    Calculate confidence score for labeling (0-1)
    Based on number and frequency of keywords
    """
    matched_keywords = find_keywords_in_text(text)
    keyword_count = len(matched_keywords)
    
    if keyword_count == 0:
        return 0.0
    
    # Confidence increases with more keywords
    confidence = min(keyword_count / 5, 1.0)  # Max at 5 keywords
    return confidence

def validate_labels(df, log_file):
    """Validate label distribution and quality"""
    disease_count = (df['label'] == 1).sum()
    normal_count = (df['label'] == 0).sum()
    total = len(df)
    
    log_message(log_file, "\nLabel Distribution:")
    log_message(log_file, f"  Disease-related: {disease_count} ({disease_count/total*100:.1f}%)")
    log_message(log_file, f"  Normal: {normal_count} ({normal_count/total*100:.1f}%)")
    
    # Check if distribution is reasonable
    disease_ratio = disease_count / total
    
    if disease_ratio < 0.05:
        log_message(log_file, f"  ⚠️  WARNING: Very few disease tweets ({disease_ratio*100:.1f}%)")
    elif disease_ratio > 0.8:
        log_message(log_file, f"  ⚠️  WARNING: Too many disease tweets ({disease_ratio*100:.1f}%)")
    else:
        log_message(log_file, "  ✓ Distribution looks reasonable")
    
    return disease_count, normal_count

def analyze_keyword_usage(df, log_file):
    """Analyze which keywords are most common"""
    log_message(log_file, "\nAnalyzing keyword usage...")
    
    all_keywords = []
    for text in df[df['label'] == 1]['clean_text']:
        all_keywords.extend(find_keywords_in_text(text))
    
    keyword_counts = Counter(all_keywords)
    most_common = keyword_counts.most_common(10)
    
    log_message(log_file, "\nTop 10 most common keywords:")
    for keyword, count in most_common:
        log_message(log_file, f"  {keyword}: {count} occurrences")

def create_balanced_dataset(df, log_file):
    """
    Create balanced dataset with equal disease and normal samples
    """
    disease_count = (df['label'] == 1).sum()
    normal_count = (df['label'] == 0).sum()
    
    if disease_count == 0 or normal_count == 0:
        log_message(log_file, "\n⚠️  Cannot create balanced dataset - missing one class")
        return df
    
    min_count = min(disease_count, normal_count)
    
    log_message(log_file, f"\nCreating balanced dataset...")
    log_message(log_file, f"  Using {min_count} samples per class")
    
    # Sample from each class
    disease_tweets = df[df['label'] == 1].sample(n=min(min_count, disease_count), 
                                                  random_state=42)
    normal_tweets = df[df['label'] == 0].sample(n=min(min_count, normal_count), 
                                                 random_state=42)
    
    # Combine and shuffle
    df_balanced = pd.concat([disease_tweets, normal_tweets])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    log_message(log_file, f"  ✓ Balanced dataset: {len(df_balanced)} total samples")
    log_message(log_file, f"    - Disease: {(df_balanced['label'] == 1).sum()}")
    log_message(log_file, f"    - Normal: {(df_balanced['label'] == 0).sum()}")
    
    return df_balanced

def show_sample_labels(df, log_file):
    """Display sample labeled tweets for verification"""
    log_message(log_file, "\nSample Disease-Related Tweets:")
    disease_samples = df[df['label'] == 1].head(3)
    for idx, (_, row) in enumerate(disease_samples.iterrows(), 1):
        keywords = find_keywords_in_text(row['clean_text'])
        log_message(log_file, f"  [{idx}] {row['clean_text'][:80]}...")
        log_message(log_file, f"      Keywords: {', '.join(keywords)}")
    
    log_message(log_file, "\nSample Normal Tweets:")
    normal_samples = df[df['label'] == 0].head(3)
    for idx, (_, row) in enumerate(normal_samples.iterrows(), 1):
        log_message(log_file, f"  [{idx}] {row['clean_text'][:80]}...")

def save_results(df, output_file, log_file):
    """Save labeled dataset with metadata"""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save labeled data
    df.to_csv(output_file, index=False)
    
    # Save metadata
    metadata = {
        'total_samples': len(df),
        'disease_samples': (df['label'] == 1).sum(),
        'normal_samples': (df['label'] == 0).sum(),
        'keyword_threshold': KEYWORD_THRESHOLD,
        'num_keywords': len(DISEASE_KEYWORDS),
        'timestamp': datetime.now().isoformat()
    }
    
    metadata_file = output_file.replace('.csv', '_metadata.txt')
    with open(metadata_file, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    log_message(log_file, f"\n✓ Labeled data saved to: {output_file}")
    log_message(log_file, f"✓ Metadata saved to: {metadata_file}")

def label_dataset(input_file='data/processed/tweets_cleaned.csv',
                 output_file='data/processed/tweets_labeled.csv'):
    """
    Main labeling function
    """
    log_file = setup_logging()
    
    print_header("DATA LABELING")
    log_message(log_file, "Starting labeling process")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        log_message(log_file, f"✗ ERROR: Input file not found: {input_file}")
        log_file.close()
        sys.exit(1)
    
    # Load cleaned data
    log_message(log_file, f"\n[1/6] Loading cleaned dataset...")
    df = pd.read_csv(input_file)
    log_message(log_file, f"    ✓ Loaded {len(df)} cleaned tweets")
    
    # Apply labeling
    log_message(log_file, "\n[2/6] Labeling tweets...")
    df['label'] = df['clean_text'].apply(label_tweet)
    df['confidence'] = df['clean_text'].apply(calculate_confidence)
    log_message(log_file, "    ✓ Labeling complete")
    
    # Validate labels
    log_message(log_file, "\n[3/6] Validating labels...")
    disease_count, normal_count = validate_labels(df, log_file)
    
    # Analyze keyword usage
    log_message(log_file, "\n[4/6] Analyzing keyword patterns...")
    analyze_keyword_usage(df, log_file)
    
    # Create balanced dataset
    log_message(log_file, "\n[5/6] Creating balanced dataset...")
    df_balanced = create_balanced_dataset(df, log_file)
    
    # Show samples
    log_message(log_file, "\n[6/6] Verification samples...")
    show_sample_labels(df_balanced, log_file)
    
    # Save results
    save_results(df_balanced, output_file, log_file)
    
    print_header("LABELING COMPLETE!")
    log_message(log_file, "Labeling process completed successfully")
    
    log_file.close()

if __name__ == "__main__":
    try:
        label_dataset()
    except Exception as e:
        print(f"\n✗ CRITICAL ERROR: {e}")
        sys.exit(1)
