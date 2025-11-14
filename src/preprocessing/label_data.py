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

OUTBREAK_INDICATORS = {
    'community_spread': [
        'everyone', 'everybody', 'whole', 'entire', 'many people',
        'lots of people', 'several people', 'multiple people',
        'all my', 'spreading', 'spread', 'outbreak', 'epidemic',
        'going around', 'community', 'neighborhood', 'school',
        'office', 'workplace', 'family members'
    ],
    
    'severity': [
        'hospital', 'hospitalized', 'emergency', 'icu', 'critical',
        'severe', 'serious', 'dying', 'death', 'deaths',
        'ambulance', 'admitted', 'er visit', 'intensive care'
    ],
    
    'rapid_increase': [
        'sudden', 'suddenly', 'rapidly', 'fast', 'quickly',
        'increase', 'rising', 'surge', 'spike', 'jump',
        'more and more', 'getting worse', 'spreading fast'
    ],
    
    'geographic': [
        'in my area', 'in my city', 'in my town', 'around here',
        'local', 'nearby', 'region', 'county', 'state'
    ],
    
    'multiple_cases': [
        'another', 'more', 'additional', 'new cases',
        'confirmed cases', 'reported cases', 'total cases',
        'case count', 'number of cases'
    ],
    
    'official_concern': [
        'health department', 'cdc', 'who', 'health officials',
        'authorities', 'warning', 'alert', 'advisory',
        'public health', 'investigation', 'monitoring'
    ]
}

# Individual symptom keywords (NOT outbreak indicators by themselves)
INDIVIDUAL_SYMPTOMS = [
    'fever', 'cough', 'headache', 'pain', 'sore throat',
    'runny nose', 'tired', 'fatigue', 'nausea', 'vomiting',
    'diarrhea', 'chills', 'ache', 'sick', 'ill', 'unwell'
]

# Configuration
OUTBREAK_THRESHOLD = 3  # Minimum outbreak indicators required
MIN_TEXT_LENGTH = 10  # Minimum words for valid labeling
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
    print("\n" + "="*70)
    print(" "*((70 - len(title))//2) + title)
    print("="*70)

def find_outbreak_indicators(text):
    """
    Find outbreak indicators in text
    Returns: dict with categories and matched indicators
    """
    if pd.isna(text) or text == "":
        return {}
    
    text_lower = text.lower()
    matched = {}
    
    for category, indicators in OUTBREAK_INDICATORS.items():
        matched_in_category = [ind for ind in indicators if ind in text_lower]
        if matched_in_category:
            matched[category] = matched_in_category
    
    return matched

def has_individual_symptoms_only(text):
    """
    Check if text only mentions individual symptoms without outbreak context
    Examples: "I have a fever", "my head hurts", "feeling sick today"
    """
    if pd.isna(text) or text == "":
        return False
    
    text_lower = text.lower()
    
    # Personal pronouns indicate individual experience
    personal_indicators = ['i have', 'i am', 'i feel', 'i got', "i'm", 
                          'my ', 'me ', 'feeling ', 'got a', 'today', 'yesterday', 'think i', 
                          'might be', 'maybe']
    
    has_personal = any(ind in text_lower for ind in personal_indicators)
    has_symptom = any(sym in text_lower for sym in INDIVIDUAL_SYMPTOMS)

    if len(text_lower.split()) < 5:
        return True
    
    return has_personal and has_symptom

def label_tweet(text):
    """
    Label tweet as outbreak-related (1) or normal (0)
    
    OUTBREAK (1) if:
    - Has multiple outbreak indicators (community spread, severity, etc.)
    - References multiple people being sick
    - Mentions official health concerns
    - Geographic clustering signals
    
    NORMAL (0) if:
    - Individual symptom mention only
    - Personal health experience
    - General health information
    - Medical advice or tips
    """
    if pd.isna(text) or text == "":
        return 0
    
    # Check for individual symptom mentions only
    if has_individual_symptoms_only(text):
        return 0  # Normal - just personal illness
    
    # Find outbreak indicators
    outbreak_indicators = find_outbreak_indicators(text)
    
    if not outbreak_indicators:
        return 0  # No outbreak signals
    
    # Count total outbreak indicators
    total_indicators = sum(len(inds) for inds in outbreak_indicators.values())
    
    # Strong outbreak signals (any one is sufficient)
    strong_signals = ['community_spread', 'official_concern', 'multiple_cases']
    has_strong_signal = any(cat in outbreak_indicators for cat in strong_signals)
    
    # Label as outbreak if:
    # 1. Has strong signal, OR
    # 2. Has multiple outbreak indicators (>= threshold)
    if has_strong_signal or total_indicators >= OUTBREAK_THRESHOLD:
        return 1
    
    return 0

def calculate_confidence(text):
    """
    Calculate confidence score for labeling (0-1)
    Based on number and strength of outbreak indicators
    """
    if pd.isna(text) or text == "":
        return 0.0
    
    outbreak_indicators = find_outbreak_indicators(text)
    
    if not outbreak_indicators:
        # High confidence it's normal if only individual symptoms
        if has_individual_symptoms_only(text):
            return 0.9
        return 0.7
    
    # Calculate confidence based on indicator strength
    strong_signals = ['community_spread', 'official_concern', 'multiple_cases']
    has_strong = any(cat in outbreak_indicators for cat in strong_signals)
    
    total_indicators = sum(len(inds) for inds in outbreak_indicators.values())
    
    if has_strong:
        confidence = 0.8 + min(total_indicators * 0.05, 0.2)
    else:
        confidence = 0.6 + min(total_indicators * 0.1, 0.3)
    
    return min(confidence, 1.0)

def get_label_explanation(text):
    """
    Generate explanation for why a tweet was labeled as outbreak or normal
    Useful for validation and debugging
    """
    if pd.isna(text) or text == "":
        return "Empty text"
    
    if has_individual_symptoms_only(text):
        return "Individual symptom mention only"
    
    outbreak_indicators = find_outbreak_indicators(text)
    
    if not outbreak_indicators:
        return "No outbreak indicators found"
    
    explanations = []
    for category, indicators in outbreak_indicators.items():
        explanations.append(f"{category}: {', '.join(indicators[:3])}")
    
    return " | ".join(explanations)

def validate_labels(df, log_file):
    """Validate label distribution and quality"""
    outbreak_count = (df['label'] == 1).sum()
    normal_count = (df['label'] == 0).sum()
    total = len(df)
    
    log_message(log_file, "\n📊 Label Distribution:")
    log_message(log_file, f"  🔴 Outbreak-related: {outbreak_count} ({outbreak_count/total*100:.1f}%)")
    log_message(log_file, f"  🟢 Normal health: {normal_count} ({normal_count/total*100:.1f}%)")
    
    # Check if distribution is reasonable for outbreak detection
    outbreak_ratio = outbreak_count / total if total > 0 else 0
    
    if outbreak_ratio < 0.05:
        log_message(log_file, f"  ⚠️  Very few outbreak tweets ({outbreak_ratio*100:.1f}%) - may need more data")
    elif outbreak_ratio > 0.60:
        log_message(log_file, f"  ⚠️  Too many outbreak tweets ({outbreak_ratio*100:.1f}%) - check labeling logic")
    else:
        log_message(log_file, "  ✓ Distribution looks reasonable for outbreak detection")
    
    # Expected: 20-40% outbreak-related for realistic dataset
    if 0.20 <= outbreak_ratio <= 0.40:
        log_message(log_file, "  ✓ Ratio matches expected outbreak detection patterns")
    
    return outbreak_count, normal_count

def analyze_indicator_usage(df, log_file):
    """Analyze which outbreak indicators are most common"""
    log_message(log_file, "\n🔍 Analyzing outbreak indicator patterns...")
    
    outbreak_tweets = df[df['label'] == 1]
    
    category_counts = {}
    for category in OUTBREAK_INDICATORS.keys():
        category_counts[category] = 0
    
    for text in outbreak_tweets['clean_text']:
        indicators = find_outbreak_indicators(text)
        for category in indicators.keys():
            category_counts[category] += 1
    
    log_message(log_file, "\n📈 Outbreak Indicator Categories:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            percentage = (count / len(outbreak_tweets) * 100) if len(outbreak_tweets) > 0 else 0
            log_message(log_file, f"  {category}: {count} tweets ({percentage:.1f}%)")

def show_sample_labels(df, log_file):
    """Display sample labeled tweets for verification"""
    log_message(log_file, "\n🔴 Sample OUTBREAK-Related Tweets:")
    outbreak_samples = df[df['label'] == 1].head(5)
    
    if len(outbreak_samples) == 0:
        log_message(log_file, "  (No outbreak tweets found)")
    else:
        for idx, (_, row) in enumerate(outbreak_samples.iterrows(), 1):
            explanation = get_label_explanation(row['clean_text'])
            log_message(log_file, f"\n  [{idx}] Text: {row['clean_text'][:100]}...")
            log_message(log_file, f"      Reason: {explanation}")
            log_message(log_file, f"      Confidence: {row['confidence']:.2f}")
    
    log_message(log_file, "\n🟢 Sample NORMAL Health Tweets:")
    normal_samples = df[df['label'] == 0].head(5)
    
    if len(normal_samples) == 0:
        log_message(log_file, "  (No normal tweets found)")
    else:
        for idx, (_, row) in enumerate(normal_samples.iterrows(), 1):
            log_message(log_file, f"\n  [{idx}] Text: {row['clean_text'][:100]}...")
            log_message(log_file, f"      Confidence: {row['confidence']:.2f}")

def create_balanced_dataset(df, log_file, target_ratio=0.35):
    """
    Create dataset with target ratio of outbreak tweets
    For outbreak detection: ~35% outbreak, ~65% normal is realistic
    """
    outbreak_count = (df['label'] == 1).sum()
    normal_count = (df['label'] == 0).sum()
    
    if outbreak_count == 0 or normal_count == 0:
        log_message(log_file, "\n⚠️  Cannot balance - missing one class")
        return df
    
    log_message(log_file, f"\n⚖️  Creating dataset with {target_ratio*100:.0f}% outbreak tweets...")
    
    # Calculate desired counts
    if outbreak_count / (outbreak_count + normal_count) < target_ratio:
        # Limited by outbreak tweets
        final_outbreak = outbreak_count
        final_normal = int(outbreak_count * (1 - target_ratio) / target_ratio)
        final_normal = min(final_normal, normal_count)
    else:
        # Limited by normal tweets  
        final_normal = normal_count
        final_outbreak = int(normal_count * target_ratio / (1 - target_ratio))
        final_outbreak = min(final_outbreak, outbreak_count)
    
    log_message(log_file, f"  Target: {final_outbreak} outbreak + {final_normal} normal")
    
    # Sample from each class
    outbreak_tweets = df[df['label'] == 1].sample(n=final_outbreak, random_state=42)
    normal_tweets = df[df['label'] == 0].sample(n=final_normal, random_state=42)
    
    # Combine and shuffle
    df_balanced = pd.concat([outbreak_tweets, normal_tweets])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    actual_ratio = (df_balanced['label'] == 1).sum() / len(df_balanced)
    log_message(log_file, f"  ✓ Created: {len(df_balanced)} samples")
    log_message(log_file, f"    - Outbreak: {(df_balanced['label'] == 1).sum()} ({actual_ratio*100:.1f}%)")
    log_message(log_file, f"    - Normal: {(df_balanced['label'] == 0).sum()} ({(1-actual_ratio)*100:.1f}%)")
    
    return df_balanced

def save_results(df, output_file, log_file):
    """Save labeled dataset with metadata"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Add explanation column for debugging
    df['label_reason'] = df['clean_text'].apply(get_label_explanation)
    
    # Save labeled data
    df.to_csv(output_file, index=False)
    
    # Save metadata
    metadata = {
        'total_samples': len(df),
        'outbreak_samples': (df['label'] == 1).sum(),
        'normal_samples': (df['label'] == 0).sum(),
        'outbreak_ratio': f"{(df['label'] == 1).sum() / len(df) * 100:.1f}%",
        'threshold': OUTBREAK_THRESHOLD,
        'labeling_approach': 'Outbreak indicators (not individual symptoms)',
        'timestamp': datetime.now().isoformat()
    }
    
    metadata_file = output_file.replace('.csv', '_metadata.txt')
    with open(metadata_file, 'w') as f:
        f.write("OUTBREAK DETECTION LABELING METADATA\n")
        f.write("="*50 + "\n\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nOUTBREAK INDICATORS:\n")
        for category, indicators in OUTBREAK_INDICATORS.items():
            f.write(f"\n{category}:\n")
            f.write(f"  {', '.join(indicators[:10])}\n")
    
    log_message(log_file, f"\n Results saved:")
    log_message(log_file, f"  ✓ Data: {output_file}")
    log_message(log_file, f"  ✓ Metadata: {metadata_file}")

def label_dataset(input_file='data/processed/tweets_cleaned.csv',
                 output_file='data/processed/tweets_labeled.csv'):
    """Main labeling function for OUTBREAK DETECTION"""
    
    log_file = setup_logging()
    
    print_header("OUTBREAK DETECTION LABELING")
    log_message(log_file, " Starting outbreak-focused labeling process")
    log_message(log_file, "Note: 'I have fever' = NORMAL | 'Everyone sick' = OUTBREAK")
    
    if not os.path.exists(input_file):
        log_message(log_file, f"✗ ERROR: Input file not found: {input_file}")
        log_file.close()
        sys.exit(1)
    
    # Load data
    log_message(log_file, f"\n[1/7]  Loading cleaned dataset...")
    df = pd.read_csv(input_file)
    log_message(log_file, f"    ✓ Loaded {len(df)} tweets")
    
    # Apply outbreak-focused labeling
    log_message(log_file, "\n[2/7]   Applying outbreak detection labels...")
    df['label'] = df['clean_text'].apply(label_tweet)
    df['confidence'] = df['clean_text'].apply(calculate_confidence)
    log_message(log_file, "    ✓ Labeling complete")
    
    # Validate labels
    log_message(log_file, "\n[3/7]  Validating labels...")
    outbreak_count, normal_count = validate_labels(df, log_file)
    
    # Analyze indicators
    log_message(log_file, "\n[4/7]  Analyzing indicator patterns...")
    analyze_indicator_usage(df, log_file)
    
    # Show samples for verification
    log_message(log_file, "\n[5/7]  Sample verification...")
    show_sample_labels(df, log_file)
    
    # Create balanced dataset
    log_message(log_file, "\n[6/7]   Balancing dataset...")
    df_balanced = create_balanced_dataset(df, log_file, target_ratio=0.35)
    
    # Save results
    log_message(log_file, "\n[7/7]  Saving results...")
    save_results(df_balanced, output_file, log_file)
    
    print_header("✅ LABELING COMPLETE")
    log_message(log_file, " Outbreak detection labeling completed successfully!")
    log_message(log_file, f" Output: {output_file}")
    
    log_file.close()

if __name__ == "__main__":
    try:
        label_dataset()
    except Exception as e:
        print(f"\n✗ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
