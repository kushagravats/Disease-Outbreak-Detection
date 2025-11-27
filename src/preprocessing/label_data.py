"""
FINAL BALANCED Data Labeling Module for Outbreak Detection
Correctly handles all edge cases

Target: 20-35% outbreak tweets for effective training
"""

import pandas as pd
import os
import sys
from datetime import datetime
import re

# OUTBREAK indicators (community-level signals)
OUTBREAK_INDICATORS = {
    'community_spread': [
        'everyone', 'everybody', 'whole school', 'entire office', 
        'many people', 'lots of people', 'several people', 'multiple people',
        'all my', 'spreading through', 'going around', 'everywhere',
        'outbreak', 'epidemic', 'pandemic', 'community', 'neighborhood', 
        'widespread', 'cluster'
    ],
    
    'severity': [
        'hospitalized', 'hospital bed', 'emergency room', 'icu', 'intensive care',
        'critical', 'severe', 'death toll', 'died from', 'fatal',
        'ambulance called', 'admitted to', 'serious condition'
    ],
    
    'rapid_increase': [
        'sudden outbreak', 'rapidly spreading', 'spike in cases',
        'surge of', 'increasing cases', 'rise in cases', 'more cases',
        'spreading fast', 'growing number', 'exponential'
    ],
    
    'official_concern': [
        'health department', 'cdc', 'who', 'health officials',
        'public health', 'authorities', 'warning', 'alert',
        'investigation', 'monitoring', 'health alert'
    ],
    
    'multiple_cases': [
        'confirmed cases', 'reported cases', 'new cases', 'total cases',
        'dozens of', 'hundreds of', 'many infected', 'multiple confirmed'
    ],
    
    'geographic': [
        'in my area', 'in my city', 'in my town', 'around here',
        'local outbreak', 'regional', 'state-wide', 'county'
    ]
}

# Words that indicate individual experience (NOT outbreak)
INDIVIDUAL_INDICATORS = [
    'i have', 'i feel', 'i am', 'i got', 'i think i', "i'm",
    'my head', 'my throat', 'my stomach', 'feeling sick',
    'today i', 'yesterday i'
]

# Explicit negation phrases (DEFINITELY not outbreak)
NEGATION_PHRASES = [
    'no one else', 'nobody else', 'no one around me', 'no one near me',
    'just me', 'only me', 'alone with', 'by myself'
]

def setup_logging():
    """Initialize logging"""
    os.makedirs('logs', exist_ok=True)
    return open('logs/labeling_final.log', 'a', encoding='utf-8')

def log_message(log_file, message):
    """Write to log"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}\n"
    print(message)
    log_file.write(log_entry)
    log_file.flush()

def has_negation(text):
    """Check if text explicitly negates others being sick"""
    if pd.isna(text):
        return False
    
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in NEGATION_PHRASES)

def has_individual_context(text):
    """Check if text has individual illness context (not outbreak)"""
    if pd.isna(text):
        return False
    
    text_lower = text.lower()
    
    # Check for personal pronouns
    has_personal = any(phrase in text_lower for phrase in INDIVIDUAL_INDICATORS)
    
    # Short text with personal context is likely individual
    word_count = len(text_lower.split())
    if has_personal and word_count < 10:
        return True
    
    return False

def count_outbreak_signals(text):
    """
    Count outbreak indicators - FIXED to avoid false positives
    """
    if pd.isna(text):
        return 0, [], {}
    
    text_lower = text.lower()
    matched_categories = []
    all_matches = {}
    
    # CRITICAL FIX: Don't match generic words like "ill", "sick", "hospital"
    # without proper context
    for category, indicators in OUTBREAK_INDICATORS.items():
        matches = []
        
        for indicator in indicators:
            # For single generic words, require them to be in specific phrases
            if indicator in ['ill', 'sick', 'hospital', 'deaths']:
                # Skip these - too generic, cause false positives
                continue
            
            if indicator in text_lower:
                matches.append(indicator)
        
        if matches:
            matched_categories.append(category)
            all_matches[category] = matches
    
    return len(matched_categories), matched_categories, all_matches

def label_tweet(text):
    """
    FINAL Labeling Logic - Correctly handles all cases
    
    Returns 0 (NORMAL) if:
    1. Explicitly negates others being sick ("no one else sick")
    2. Individual context with no outbreak indicators
    3. No outbreak indicators at all
    
    Returns 1 (OUTBREAK) if:
    1. Has outbreak indicators AND
    2. NOT explicitly individual
    """
    if pd.isna(text) or text.strip() == "":
        return 0
    
    # PRIORITY 1: Check for explicit negation
    if has_negation(text):
        return 0  # Definitely normal
    
    # PRIORITY 2: Count outbreak signals
    signal_count, categories, matches = count_outbreak_signals(text)
    
    # If no outbreak signals, it's normal
    if signal_count == 0:
        return 0
    
    # PRIORITY 3: If has outbreak signals AND individual context
    # Need to decide based on strength of outbreak signals
    if has_individual_context(text):
        # If only weak signals, label as normal
        if signal_count < 2:
            return 0
    
    # Has outbreak indicators and not explicitly individual
    return 1

def calculate_confidence(text):
    """Calculate confidence score"""
    if pd.isna(text):
        return 0.0
    
    # High confidence for explicit negation
    if has_negation(text):
        return 0.95
    
    signal_count, categories, _ = count_outbreak_signals(text)
    
    # No signals = medium confidence normal
    if signal_count == 0:
        return 0.75
    
    # More signals = higher confidence
    strong_categories = ['community_spread', 'official_concern', 'rapid_increase']
    strong_count = sum(1 for cat in categories if cat in strong_categories)
    
    base_confidence = 0.65
    confidence = base_confidence + (signal_count * 0.08) + (strong_count * 0.10)
    
    return min(confidence, 0.95)

def get_label_reason(text):
    """Get human-readable reason"""
    if pd.isna(text):
        return "Empty text"
    
    if has_negation(text):
        return "Explicitly states others not sick"
    
    signal_count, categories, matches = count_outbreak_signals(text)
    
    if signal_count == 0:
        if has_individual_context(text):
            return "Individual illness mention only"
        return "No outbreak indicators"
    
    # Build explanation
    explanations = []
    for cat in categories[:3]:
        indicators = matches[cat][:2]
        explanations.append(f"{cat}: {', '.join(indicators)}")
    
    return " | ".join(explanations)

def validate_critical_cases(df, log_file):
    """Validate critical test cases with CORRECT expectations"""
    log_message(log_file, "\n Validating Critical Test Cases:")
    
    test_cases = [
        # Explicit negation - should be NORMAL
        ("I have chills and headache but no one around me is sick", 0, "Explicit negation"),
        ("Just me sick, nobody else", 0, "Explicit individual"),
        
        # Individual symptoms only - should be NORMAL  
        ("I have fever and cough", 0, "Individual symptoms, no context"),
        ("Feeling sick today", 0, "Personal experience only"),
        ("I think I have flu", 0, "Individual suspicion"),
        ("My head hurts so bad", 0, "Personal symptom"),
        
        # Clear outbreak indicators - should be OUTBREAK
        ("Everyone in my office has the flu", 1, "Community spread"),
        ("Multiple people hospitalized with same symptoms", 1, "Severity + community"),
        ("Outbreak reported in my neighborhood", 1, "Official outbreak mention"),
        ("Health department investigating illness cluster", 1, "Official concern"),
        ("Going around school", 1, "Community spread"),
        ("Dozens of confirmed cases in my area", 1, "Multiple cases + geographic"),
        ("Rapidly spreading through the community", 1, "Rapid increase + community"),
    ]
    
    passed = 0
    failed = 0
    
    for text, expected_label, reasoning in test_cases:
        actual_label = label_tweet(text)
        confidence = calculate_confidence(text)
        reason = get_label_reason(text)
        
        if actual_label == expected_label:
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"
            failed += 1
        
        label_str = "OUTBREAK" if actual_label == 1 else "NORMAL"
        expected_str = "OUTBREAK" if expected_label == 1 else "NORMAL"
        
        log_message(log_file, f"\n  {status}")
        log_message(log_file, f"    Text: '{text}'")
        log_message(log_file, f"    Expected: {expected_str} | Got: {label_str}")
        log_message(log_file, f"    Why: {reasoning}")
        log_message(log_file, f"    Confidence: {confidence:.2f}")
        log_message(log_file, f"    Reason: {reason}")
    
    log_message(log_file, f"\n  Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        log_message(log_file, "  ALL TEST CASES PASSED!")
    else:
        log_message(log_file, f"  {failed} test cases failed")
    
    return failed == 0

def validate_labels(df, log_file):
    """Validate overall label distribution"""
    outbreak_count = (df['label'] == 1).sum()
    normal_count = (df['label'] == 0).sum()
    total = len(df)
    
    log_message(log_file, "\n Label Distribution:")
    log_message(log_file, f"   Outbreak: {outbreak_count} ({outbreak_count/total*100:.1f}%)")
    log_message(log_file, f"   Normal: {normal_count} ({normal_count/total*100:.1f}%)")
    
    outbreak_ratio = outbreak_count / total if total > 0 else 0
    
    if outbreak_ratio < 0.10:
        log_message(log_file, f"   WARNING: Too few outbreaks ({outbreak_ratio*100:.1f}%)")
        log_message(log_file, "     Models need at least 10% outbreak samples")
    elif outbreak_ratio < 0.15:
        log_message(log_file, f"   Low outbreak ratio ({outbreak_ratio*100:.1f}%)")
        log_message(log_file, "     20-35% would be better")
    elif outbreak_ratio > 0.50:
        log_message(log_file, f"   Too many outbreaks ({outbreak_ratio*100:.1f}%)")
    else:
        log_message(log_file, "   Excellent distribution for training!")
    
    return outbreak_count, normal_count

def show_samples(df, log_file, n=10):
    """Show sample labels"""
    log_message(log_file, "\n Sample OUTBREAK Tweets:")
    outbreak_samples = df[df['label'] == 1].head(n)
    
    for idx, (_, row) in enumerate(outbreak_samples.iterrows(), 1):
        log_message(log_file, f"\n  [{idx}] {row['clean_text'][:100]}...")
        log_message(log_file, f"      Reason: {row['label_reason']}")
        log_message(log_file, f"      Confidence: {row['confidence']:.2f}")
    
    log_message(log_file, "\n Sample NORMAL Tweets:")
    normal_samples = df[df['label'] == 0].head(n)
    
    for idx, (_, row) in enumerate(normal_samples.iterrows(), 1):
        log_message(log_file, f"\n  [{idx}] {row['clean_text'][:100]}...")
        log_message(log_file, f"      Reason: {row['label_reason']}")
        log_message(log_file, f"      Confidence: {row['confidence']:.2f}")

def create_balanced_dataset(df, log_file):
    """Create balanced dataset - 70% normal, 30% outbreak"""
    outbreak_count = (df['label'] == 1).sum()
    normal_count = (df['label'] == 0).sum()
    
    log_message(log_file, f"\n Creating Balanced Dataset...")
    log_message(log_file, f"  Original: {outbreak_count} outbreak, {normal_count} normal")
    
    if outbreak_count < 500:
        log_message(log_file, f"   Few outbreak samples ({outbreak_count})")
        log_message(log_file, "     Using all data")
        return df
    
    # Target: 30% outbreak
    outbreak_tweets = df[df['label'] == 1]
    target_normal = int(outbreak_count * 2.33)  # 30/70 ratio
    target_normal = min(target_normal, normal_count)
    
    normal_tweets = df[df['label'] == 0].sample(n=target_normal, random_state=42)
    
    df_balanced = pd.concat([outbreak_tweets, normal_tweets])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    actual_ratio = (df_balanced['label'] == 1).sum() / len(df_balanced)
    
    log_message(log_file, f"\n   Balanced Dataset:")
    log_message(log_file, f"     Total: {len(df_balanced)}")
    log_message(log_file, f"     Outbreak: {(df_balanced['label']==1).sum()} ({actual_ratio*100:.1f}%)")
    log_message(log_file, f"     Normal: {(df_balanced['label']==0).sum()} ({(1-actual_ratio)*100:.1f}%)")
    
    return df_balanced

def label_dataset(input_file='data/processed/tweets_cleaned.csv',
                 output_file='data/processed/tweets_labeled.csv'):
    """Main labeling function - FINAL VERSION"""
    
    log_file = setup_logging()
    
    print("\n" + "="*70)
    print(" "*15 + "FINAL OUTBREAK DETECTION LABELING")
    print("="*70)
    
    log_message(log_file, "🔧 Starting FINAL labeling process")
    log_message(log_file, "Goal: Accurate labels with good training distribution")
    
    if not os.path.exists(input_file):
        log_message(log_file, f"✗ ERROR: Input file not found: {input_file}")
        log_file.close()
        sys.exit(1)
    
    log_message(log_file, f"\n[1/6] Loading data...")
    df = pd.read_csv(input_file)
    log_message(log_file, f"  ✓ Loaded {len(df)} tweets")
    
    log_message(log_file, "\n[2/6] Validating labeling logic...")
    all_passed = validate_critical_cases(df, log_file)
    
    if not all_passed:
        log_message(log_file, "\n Warning: Some test cases failed")
        log_message(log_file, "Review the logic if needed")
    
    log_message(log_file, "\n[3/6] Applying labels...")
    df['label'] = df['clean_text'].apply(label_tweet)
    df['confidence'] = df['clean_text'].apply(calculate_confidence)
    df['label_reason'] = df['clean_text'].apply(get_label_reason)
    log_message(log_file, "  ✓ Labeling complete")
    
    log_message(log_file, "\n[4/6] Validating distribution...")
    outbreak_count, normal_count = validate_labels(df, log_file)
    
    log_message(log_file, "\n[5/6] Reviewing samples...")
    show_samples(df, log_file, n=8)
    
    log_message(log_file, "\n[6/6] Creating balanced dataset...")
    df_balanced = create_balanced_dataset(df, log_file)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_balanced.to_csv(output_file, index=False)
    
    final_outbreak = (df_balanced['label']==1).sum()
    final_normal = (df_balanced['label']==0).sum()
    final_ratio = final_outbreak / len(df_balanced)
    
    log_message(log_file, f"\n SUCCESS!")
    log_message(log_file, f"  Saved: {output_file}")
    log_message(log_file, f"  Total: {len(df_balanced)}")
    log_message(log_file, f"  Outbreak: {final_outbreak} ({final_ratio*100:.1f}%)")
    log_message(log_file, f"  Normal: {final_normal} ({(1-final_ratio)*100:.1f}%)")
    
    print("\n" + "="*70)
    if all_passed and 0.20 <= final_ratio <= 0.40:
        print(" LABELING COMPLETE - Perfect for training!")
    elif final_ratio < 0.15:
        print(" LABELING COMPLETE - Low outbreak ratio")
    else:
        print(" LABELING COMPLETE - Ready for training")
    print("="*70)
    print(f"\n Results:")
    print(f"   • {final_outbreak} outbreak tweets ({final_ratio*100:.1f}%)")
    print(f"   • {final_normal} normal tweets ({(1-final_ratio)*100:.1f}%)")
    print(f"\n Next: Retrain your models!")
    
    log_file.close()
    return df_balanced

if __name__ == "__main__":
    try:
        label_dataset()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)