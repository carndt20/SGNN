import pandas as pd
import os
import sys
from datetime import datetime, timedelta

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from utils.file_utils import absolute_path
from utils.config_utils import get_config

def create_txs_time():
    """Create txs_time.csv with dummy timestamps"""
    print("Creating txs_time.csv with dummy timestamps...")
    
    config = get_config('path')
    
    # Read txs_features.csv first to get the exact transactions we need
    print("Reading txs_features.csv...")
    tx_features = pd.read_csv(absolute_path(config['file']['txs_features']))
    
    # Read txId2hash.csv to get the hash mapping
    print("Reading txId2hash.csv...")
    txId2hash = pd.read_csv(absolute_path(config['file']['txId2hash']))
    
    # Create a mapping of txId to txhash
    txid_to_hash = txId2hash.set_index('txId')['txhash'].to_dict()
    
    # Filter tx_features to only include transactions that have hashes
    valid_txs = tx_features[tx_features['txId'].isin(txid_to_hash.keys())]
    print(f"Found {len(valid_txs)} transactions with valid hashes out of {len(tx_features)} total transactions")
    
    # Create dummy timestamps starting from a base date
    base_date = datetime(2019, 1, 1)  # Starting from January 1, 2019
    time_step = timedelta(days=1)  # One day per transaction
    
    # Create timestamps for each valid transaction
    timestamps = [base_date + (i * time_step) for i in range(len(valid_txs))]
    
    # Create the DataFrame with valid transactions
    txs_time = pd.DataFrame({
        'txId': valid_txs['txId'],
        'timestamp': timestamps
    })
    
    # Add txhash column by mapping from txId
    txs_time['txhash'] = txs_time['txId'].map(txid_to_hash)
    
    # Ensure txhash is string type and remove any leading/trailing whitespace
    txs_time['txhash'] = txs_time['txhash'].astype(str).str.strip()
    
    # Reorder columns to match expected format (txhash first, then txId, then timestamp)
    txs_time = txs_time[['txhash', 'txId', 'timestamp']]
    
    # Save to file
    output_path = absolute_path(config['file']['txs_time'])
    print(f"Saving to {output_path}...")
    txs_time.to_csv(output_path, index=False)
    print("Done!")

if __name__ == '__main__':
    create_txs_time() 