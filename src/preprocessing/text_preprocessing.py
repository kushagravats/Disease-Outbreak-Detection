"""
Text Preprocessing Module
Cleans and prepares tweet text for ML models
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK data is downloaded
try:
    stopwords.words('english')
except:
    nltk.download('stopwords')
    nltk.download('punkt')

def clean_text(text):
    """
    Clean and normalize tweet text
    Steps:
    1. Lowercase conversion
    2. URL removal
    3. Mention removal
    4. Hashtag symbol removal
    5. Special character removal
    6. Stop word removal
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtag symbol but keep the word
    text = re.sub(r'#', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    return ' '.join(words)

def preprocess_dataset(input_file='data/processed/tweets_loaded.csv',
                      output_file='data/processed/tweets_cleaned.csv'):
    """
    Preprocess entire dataset
    """
    print("="*60)
    print("TEXT PREPROCESSING")
    print("="*60)
    
    # Load data
    df = pd.read_csv(input_file)
    print(f"\n✓ Loaded {len(df)} tweets")
    
    # Clean text
    print("\nCleaning text...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Remove very short texts (less than 10 characters)
    df = df[df['clean_text'].str.len() >= 10]
    print(f"✓ {len(df)} tweets remaining after removing short texts")
    
    # Remove duplicates
    original_count = len(df)
    df = df.drop_duplicates(subset=['clean_text'])
    print(f"✓ Removed {original_count - len(df)} duplicate tweets")
    
    # Save cleaned data
    df.to_csv(output_file, index=False)
    print(f"\n✓ Cleaned data saved to: {output_file}")
    
    # Show examples
    print("\nExample cleaned tweets:")
    for i in range(min(3, len(df))):
        print(f"\nOriginal: {df.iloc[i]['text'][:100]}...")
        print(f"Cleaned:  {df.iloc[i]['clean_text'][:100]}...")
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    
    return df

if __name__ == "__main__":
    preprocess_dataset()
