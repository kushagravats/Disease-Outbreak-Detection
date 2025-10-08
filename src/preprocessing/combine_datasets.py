"""
Combine Multiple Health News Files
Handles pipe-delimited (|) and comma-separated CSV files
Advanced data validation and quality checks included

Features:
- Multi-format support (CSV, TXT with pipe or comma delimiters)
- Data quality validation
- Duplicate detection and removal
- Statistical analysis of combined data
- Error handling and logging
"""

import pandas as pd
import os
import sys
from datetime import datetime

DATA_FOLDER = 'data/raw/'
OUTPUT_FILE = 'data/raw/health_tweets.csv'
MIN_TEXT_LENGTH = 10
MAX_TEXT_LENGTH = 5000

LOG_FILE = 'logs/combine_datasets.log'

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

def validate_dataframe(df, filename):
    """
    Validate dataframe structure and content
    Returns: (is_valid, error_message)
    """
    if df is None or df.empty:
        return False, "DataFrame is empty"
    
    if 'text' not in df.columns:
        return False, "Missing 'text' column"
    
    # Check for minimum data
    if len(df) < 10:
        return False, f"Too few records: {len(df)}"
    
    # Check text quality
    valid_texts = df['text'].notna() & (df['text'].str.len() >= MIN_TEXT_LENGTH)
    valid_ratio = valid_texts.sum() / len(df)
    
    if valid_ratio < 0.5:
        return False, f"Too many invalid texts: {valid_ratio*100:.1f}% valid"
    
    return True, "Valid"

def remove_duplicates(df, log_file):
    """Remove duplicate tweets based on text content"""
    original_len = len(df)
    df = df.drop_duplicates(subset=['text'], keep='first')
    removed = original_len - len(df)
    
    if removed > 0:
        log_message(log_file, f"  ✓ Removed {removed} duplicate tweets")
    
    return df

def clean_text_data(df, log_file):
    """Clean and validate text data"""
    log_message(log_file, "\nCleaning text data...")
    
    # Remove leading/trailing whitespace
    df['text'] = df['text'].str.strip()
    
    # Remove empty texts
    original_len = len(df)
    df = df[df['text'].notna()]
    df = df[df['text'].str.len() >= MIN_TEXT_LENGTH]
    removed = original_len - len(df)
    
    if removed > 0:
        log_message(log_file, f"  ✓ Removed {removed} invalid/empty texts")
    
    original_len = len(df)
    df = df[df['text'].str.len() <= MAX_TEXT_LENGTH]
    removed = original_len - len(df)
    
    if removed > 0:
        log_message(log_file, f"  ✓ Removed {removed} texts exceeding max length")
    
    return df

def get_text_statistics(df):
    """Calculate text statistics"""
    text_lengths = df['text'].str.len()
    
    stats = {
        'count': len(df),
        'min_length': text_lengths.min(),
        'max_length': text_lengths.max(),
        'mean_length': text_lengths.mean(),
        'median_length': text_lengths.median()
    }
    
    return stats

def print_header(title):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(" "*((70 - len(title))//2) + title)
    print("="*70)

def find_health_files(data_folder):
    """
    Find all health-related data files
    Returns: list of filenames
    """
    all_files = os.listdir(data_folder)
    health_files = []
    
    for file in all_files:
        if 'health' in file.lower() and (file.endswith('.csv') or file.endswith('.txt')):
            if file != os.path.basename(OUTPUT_FILE):
                health_files.append(file)
    
    return health_files

def read_file_with_pipe(file_path):
    """Read file with pipe delimiter"""
    try:
        df = pd.read_csv(file_path, 
                        sep='|',
                        encoding='utf-8', 
                        on_bad_lines='skip',
                        header=None,
                        names=['id', 'date', 'text'])
        return df, True
    except Exception as e:
        return None, False

def read_file_with_comma(file_path):
    """Read file with comma delimiter"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        return df, True
    except Exception as e:
        return None, False

def process_single_file(file, data_folder, log_file):
    """
    Process a single data file
    Returns: DataFrame or None
    """
    file_path = os.path.join(data_folder, file)
    size_mb = os.path.getsize(file_path) / 1024 / 1024
    
    log_message(log_file, f"\nProcessing: {file} ({size_mb:.2f} MB)")
    
    # Try pipe delimiter first
    df, success = read_file_with_pipe(file_path)
    delimiter = 'pipe'
    
    if not success or df is None:
        log_message(log_file, "  Trying comma delimiter...")
        df, success = read_file_with_comma(file_path)
        delimiter = 'comma'
    
    if not success or df is None:
        log_message(log_file, f"  ✗ Failed to read file")
        return None
    
    # Validate dataframe
    is_valid, error_msg = validate_dataframe(df, file)
    if not is_valid:
        log_message(log_file, f"  ✗ Validation failed: {error_msg}")
        return None
    
    # Add metadata
    source_name = file.replace('_health.csv', '').replace('_health.txt', '').upper()
    df['source'] = source_name
    df['file_name'] = file
    
    # Log success
    log_message(log_file, f"  ✓ Loaded {len(df)} records using {delimiter} delimiter")
    log_message(log_file, f"  ✓ Columns: {list(df.columns)}")
    
    # Show sample
    if len(df) > 0:
        sample_text = df['text'].iloc[0][:60]
        log_message(log_file, f"  Sample: {sample_text}...")
    
    return df

def main():
    """Main execution function"""
    # Setup logging
    log_file = setup_logging()
    
    print_header("COMBINING DATASETS")
    log_message(log_file, "Starting dataset combination process")
    
    # Ensure data folder exists
    if not os.path.exists(DATA_FOLDER):
        log_message(log_file, f"✗ ERROR: Data folder not found: {DATA_FOLDER}")
        log_file.close()
        sys.exit(1)
    
    # Find health files
    log_message(log_file, "\nSearching for health data files...")
    health_files = find_health_files(DATA_FOLDER)
    
    if not health_files:
        log_message(log_file, "\n✗ ERROR: No health data files found!")
        log_file.close()
        sys.exit(1)
    
    for file in health_files:
        log_message(log_file, f"  ✓ Found: {file}")
    
    log_message(log_file, f"\n✓ Will combine {len(health_files)} file(s)")
    
    # Process all files
    print_header("READING FILES")
    all_dataframes = []
    
    for file in health_files:
        df = process_single_file(file, DATA_FOLDER, log_file)
        if df is not None:
            all_dataframes.append(df)
    
    if not all_dataframes:
        log_message(log_file, "\n✗ ERROR: No data could be loaded!")
        log_file.close()
        sys.exit(1)
    
    # Combine all dataframes
    print_header("COMBINING ALL DATASETS")
    log_message(log_file, "Merging all dataframes...")
    
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    log_message(log_file, f"✓ Initial combined size: {len(combined_df)} rows")
    
    # Data quality checks
    print_header("DATA QUALITY CHECKS")
    
    # Remove duplicates
    combined_df = remove_duplicates(combined_df, log_file)
    
    # Clean text data
    combined_df = clean_text_data(combined_df, log_file)
    
    # Calculate statistics
    log_message(log_file, "\nCalculating dataset statistics...")
    stats = get_text_statistics(combined_df)
    
    log_message(log_file, f"\nText Statistics:")
    log_message(log_file, f"  Total records: {stats['count']}")
    log_message(log_file, f"  Min length: {stats['min_length']} chars")
    log_message(log_file, f"  Max length: {stats['max_length']} chars")
    log_message(log_file, f"  Mean length: {stats['mean_length']:.1f} chars")
    log_message(log_file, f"  Median length: {stats['median_length']:.1f} chars")
    
    # Show distribution by source
    if 'source' in combined_df.columns:
        log_message(log_file, f"\nData distribution by source:")
        source_counts = combined_df['source'].value_counts()
        for source, count in source_counts.items():
            percentage = (count / len(combined_df)) * 100
            log_message(log_file, f"  {source}: {count} ({percentage:.1f}%)")
    
    # Save combined dataset
    print_header("SAVING RESULTS")
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    combined_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    
    output_size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
    
    log_message(log_file, f"\n✓ Saved to: {OUTPUT_FILE}")
    log_message(log_file, f"✓ Total records: {len(combined_df)}")
    log_message(log_file, f"✓ File size: {output_size_mb:.2f} MB")
    log_message(log_file, f"✓ Columns: {list(combined_df.columns)}")
    
    print_header("SUCCESS!")
    log_message(log_file, "Dataset combination completed successfully")
    
    # Close log file
    log_file.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ CRITICAL ERROR: {e}")
        sys.exit(1)
