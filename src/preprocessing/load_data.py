"""
Data Loading Script
Loads health-related tweets from Kaggle dataset
"""

import pandas as pd
import os

def load_raw_data(file_path='data/raw/health_tweets.csv'):
    """Load raw tweet data from CSV"""
    print("="*60)
    print("LOADING DATASET")
    print("="*60)
    
    if not os.path.exists(file_path):
        print(f"ERROR: File not found at {file_path}")
        print("Please run combine_datasets.py first to create health_tweets.csv")
        return None
    
    # Load CSV
    df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
    
    print(f"\n✓ Dataset loaded successfully!")
    print(f"Total records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    
    # Identify text column
    text_column = None
    possible_names = ['text', 'tweet', 'content', 'message', 'Text', 'Tweet']
    
    for col in possible_names:
        if col in df.columns:
            text_column = col
            break
    
    if text_column:
        print(f"\n✓ Text column identified: '{text_column}'")
        if text_column != 'text':
            df = df.rename(columns={text_column: 'text'})
            print(f"✓ Renamed to 'text'")
    else:
        print(f"\nWARNING: Could not identify text column")
        print(f"Available columns: {list(df.columns)}")
    
    # Basic statistics
    print(f"\n✓ Missing values in 'text': {df['text'].isna().sum()}")
    print(f"✓ Valid tweets: {df['text'].notna().sum()}")
    
    return df

if __name__ == "__main__":
    # Load the data
    df = load_raw_data()
    
    if df is not None:
        # Create processed folder if it doesn't exist
        os.makedirs('data/processed', exist_ok=True)
        
        # Save to processed folder
        output_path = 'data/processed/tweets_loaded.csv'
        df.to_csv(output_path, index=False)
        print(f"\n✓ Data saved to: {output_path}")
        print("\n" + "="*60)
        print("DATA LOADING COMPLETE!")
        print("="*60)
