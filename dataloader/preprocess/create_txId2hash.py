import pandas as pd
import os
import sys
import shutil

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from utils.config_utils import get_config
from utils.file_utils import absolute_path

def create_txId2hash():
    """Create txId2hash.csv by combining transaction IDs and hashes from input and output value files"""
    print("Creating txId2hash.csv...")
    
    config = get_config('path')
    
    # Read input and output value files
    print("Reading input value file...")
    in_value = pd.read_csv(absolute_path(config['file']['in_value']))
    print("Reading output value file...")
    out_value = pd.read_csv(absolute_path(config['file']['out_value']))
    
    # Extract unique txhash values
    print("Extracting unique transaction hashes...")
    txhashes = pd.concat([
        in_value[['txhash']].drop_duplicates(),
        out_value[['txhash']].drop_duplicates()
    ]).drop_duplicates()
    
    # Create sequential transaction IDs
    print("Creating transaction IDs...")
    txhashes['txId'] = range(len(txhashes))
    
    # Save to temporary file first
    temp_file = 'data/temp_txId2hash.csv'
    print(f"Saving to temporary file {temp_file}...")
    txhashes.to_csv(temp_file, index=False)
    
    # Move to final location
    final_path = absolute_path(config['file']['txId2hash'])
    print(f"Moving to final location {final_path}...")
    if os.path.exists(final_path):
        if os.path.isdir(final_path):
            shutil.rmtree(final_path)
        else:
            os.remove(final_path)
    shutil.move(temp_file, final_path)
    print("Done!")

if __name__ == '__main__':
    create_txId2hash() 