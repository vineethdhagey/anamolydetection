# scripts/simulate_stream.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.utils import load_fd002_dataset

def simulate_stream(df, batch_size=5):
    """
    Generator to simulate streaming data in batches.
    
    Args:
        df (pd.DataFrame): Input dataframe to stream.
        batch_size (int): Number of rows per batch.
        
    Yields:
        pd.DataFrame: Next batch of rows.
    """
    total_rows = len(df)
    for start_idx in range(0, total_rows, batch_size):
        yield df.iloc[start_idx:start_idx + batch_size]

def main():
    print("🌐 Loading dataset...")
    df = load_fd002_dataset()
    
    print("🔄 Simulating stream in batches of 5...")
    for i, batch in enumerate(simulate_stream(df, batch_size=5)):
        print(f"Batch {i+1}:")
        print(batch.head(), "\n")  # show first few rows of the batch

if __name__ == "__main__":
    main()
