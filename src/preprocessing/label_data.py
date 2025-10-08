"""
Data Labeling Module
Labels tweets as disease-related or normal
"""

import pandas as pd

DISEASE_KEYWORDS = [
    'flu', 'fever', 'cough', 'sick', 'illness', 'disease', 'virus',
    'symptom', 'outbreak', 'epidemic', 'pandemic', 'infection',
    'covid', 'corona', 'coronavirus', 'headache', 'pain', 
    'hospital', 'emergency', 'breathing', 'pneumonia', 'vomiting',
    'diarrhea', 'fatigue', 'spread', 'contagious', 'diagnosed',
    'tested', 'positive', 'patient', 'clinic', 'doctor'
]

def label_tweet(text):
    """
    Label tweet as disease-related (1) or normal (0)
    Requires at least 2 disease keywords to reduce false positives
    """
    if pd.isna(text) or text == "":
        return 0
    
    text_lower = text.lower()
    keyword_count = sum(1 for keyword in DISEASE_KEYWORDS if keyword in text_lower)
    
    return 1 if keyword_count >= 2 else 0

def label_dataset(input_file='data/processed/tweets_cleaned.csv',
                 output_file='data/processed/tweets_labeled.csv'):
    """
    Label entire dataset
    """
    print("="*60)
    print("DATA LABELING")
    print("="*60)
    
    # Load cleaned data
    df = pd.read_csv(input_file)
    print(f"\n✓ Loaded {len(df)} cleaned tweets")
    
    # Apply labeling
    print("\nLabeling tweets...")
    df['label'] = df['clean_text'].apply(label_tweet)
    
    # Statistics
    disease_count = (df['label'] == 1).sum()
    normal_count = (df['label'] == 0).sum()
    
    print(f"\n✓ Disease-related tweets: {disease_count} ({disease_count/len(df)*100:.1f}%)")
    print(f"✓ Normal tweets: {normal_count} ({normal_count/len(df)*100:.1f}%)")
    
    if disease_count > 0 and normal_count > 0:
        min_count = min(disease_count, normal_count)
        
        disease_tweets = df[df['label'] == 1].sample(n=min(min_count, disease_count), 
                                                      random_state=42)
        normal_tweets = df[df['label'] == 0].sample(n=min(min_count, normal_count), 
                                                     random_state=42)
        
        df_balanced = pd.concat([disease_tweets, normal_tweets])
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\n✓ Balanced dataset created: {len(df_balanced)} tweets")
        print(f"  - Disease tweets: {(df_balanced['label'] == 1).sum()}")
        print(f"  - Normal tweets: {(df_balanced['label'] == 0).sum()}")
        
        df_balanced.to_csv(output_file, index=False)
    else:
        df.to_csv(output_file, index=False)
    
    print(f"\n✓ Labeled data saved to: {output_file}")
    
    print("\nExample disease-related tweets:")
    disease_examples = df[df['label'] == 1].head(2)
    for idx, row in disease_examples.iterrows():
        print(f"  - {row['clean_text'][:80]}...")
    
    print("\nExample normal tweets:")
    normal_examples = df[df['label'] == 0].head(2)
    for idx, row in normal_examples.iterrows():
        print(f"  - {row['clean_text'][:80]}...")
    
    print("\n" + "="*60)
    print("LABELING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    label_dataset()
