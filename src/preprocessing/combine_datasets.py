"""
Combine Multiple Health News Files
Handles pipe-delimited (|) CSV files
"""

import pandas as pd
import os

print("="*70)
print(" "*20 + "COMBINING DATASETS")
print("="*70)

data_folder = 'data/raw/'

# Look for all health data files
print("\nSearching for health data files...")

all_files = os.listdir(data_folder)
health_files = []

for file in all_files:
    if 'health' in file.lower() and (file.endswith('.csv') or file.endswith('.txt')):
        if file != 'health_tweets.csv':  # Skip the output file if it exists
            health_files.append(file)
            print(f"  ✓ Found: {file}")

if not health_files:
    print("\n✗ ERROR: No health data files found!")
    exit()

print(f"\n✓ Will combine {len(health_files)} file(s)")

# Read and combine all files
print("\n" + "="*70)
print("Reading files...")
print("="*70)

all_dataframes = []

for file in health_files:
    file_path = os.path.join(data_folder, file)
    
    try:
        # First, try reading with pipe delimiter (|)
        df = pd.read_csv(file_path, 
                        sep='|',  # Use pipe as separator
                        encoding='utf-8', 
                        on_bad_lines='skip',
                        header=None,  # No header row
                        names=['id', 'date', 'text'])  # Assign column names
        
        size_mb = os.path.getsize(file_path) / 1024 / 1024
        
        print(f"\n{file}:")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Sample text: {df['text'].iloc[0][:60] if len(df) > 0 else 'N/A'}...")
        
        # Add source identifier
        source_name = file.replace('_health.csv', '').replace('_health.txt', '').upper()
        df['source'] = source_name
        
        all_dataframes.append(df)
        
    except Exception as e:
        print(f"\n✗ Error reading {file} with pipe delimiter: {e}")
        print("  Trying comma delimiter...")
        
        try:
            # Fallback to comma delimiter
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
            source_name = file.replace('_health.csv', '').replace('_health.txt', '').upper()
            df['source'] = source_name
            all_dataframes.append(df)
            print(f"  ✓ Successfully read with comma delimiter")
        except Exception as e2:
            print(f"  ✗ Failed with both delimiters: {e2}")

# Combine all dataframes
if all_dataframes:
    print("\n" + "="*70)
    print("Combining all datasets...")
    print("="*70)
    
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"\n✓ Combined dataset created!")
    print(f"  Total rows: {len(combined_df)}")
    print(f"  Columns: {list(combined_df.columns)}")
    
    # Ensure 'text' column exists
    if 'text' not in combined_df.columns:
        print("\n✗ ERROR: No 'text' column found!")
        print(f"Available columns: {list(combined_df.columns)}")
    else:
        # Show distribution by source
        if 'source' in combined_df.columns:
            print(f"\nData distribution by source:")
            print(combined_df['source'].value_counts())
        
        # Show sample data
        print(f"\nSample combined data:")
        print(f"  ID: {combined_df['id'].iloc[0] if 'id' in combined_df.columns else 'N/A'}")
        print(f"  Date: {combined_df['date'].iloc[0] if 'date' in combined_df.columns else 'N/A'}")
        print(f"  Text: {combined_df['text'].iloc[0][:80]}...")
        print(f"  Source: {combined_df['source'].iloc[0] if 'source' in combined_df.columns else 'N/A'}")
        
        # Clean up text column (remove any leading/trailing whitespace)
        combined_df['text'] = combined_df['text'].str.strip()
        
        # Remove rows with empty text
        original_len = len(combined_df)
        combined_df = combined_df[combined_df['text'].notna()]
        combined_df = combined_df[combined_df['text'].str.len() > 0]
        print(f"\n✓ Removed {original_len - len(combined_df)} rows with empty text")
        
        # Save combined dataset
        output_path = 'data/raw/health_tweets.csv'
        combined_df.to_csv(output_path, index=False, encoding='utf-8')
        
        output_size_mb = os.path.getsize(output_path) / 1024 / 1024
        
        print(f"\n" + "="*70)
        print("SUCCESS!")
        print("="*70)
        print(f"✓ Saved to: {output_path}")
        print(f"✓ Total records: {len(combined_df)}")
        print(f"✓ File size: {output_size_mb:.2f} MB")
        print(f"✓ Columns: {list(combined_df.columns)}")
        print("="*70)
    
else:
    print("\n✗ ERROR: No data could be loaded!")
