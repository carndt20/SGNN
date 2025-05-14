import pandas as pd
from utils.file_utils import absolute_path
from utils.config_utils import get_config

def create_value_edgelists():
    config = get_config('path')
    print("Reading input files...")
    
    # Read input files
    input_values = pd.read_csv(absolute_path('data/process/input_sent_value.csv'))
    addr_tx = pd.read_csv(absolute_path('data/AddrTx_edgelist.csv'))
    tx_hash = pd.read_csv(absolute_path('data/txId2hash.csv'))
    
    print("Creating AddrTxValue_edgelist.csv...")
    # Merge input_values with tx_hash to get txId
    input_values_with_id = pd.merge(input_values, tx_hash, on='txhash', how='inner')
    
    # Merge with addr_tx to get the final mapping
    addr_tx_value = pd.merge(
        addr_tx,
        input_values_with_id[['txId', 'input_address', 'input_value']],
        on=['txId', 'input_address'],
        how='inner'
    )
    
    # Save the result
    addr_tx_value.to_csv(absolute_path('data/AddrTxValue_edgelist.csv'), index=False)
    print(f"Saved AddrTxValue_edgelist.csv with {len(addr_tx_value)} rows")
    
    # Now do the same for output values
    print("\nReading output values...")
    output_values = pd.read_csv(absolute_path('data/process/output_recevied_value.csv'))
    tx_addr = pd.read_csv(absolute_path('data/TxAddr_edgelist.csv'))
    
    print("Creating TxAddrValue_edgelist.csv...")
    # Merge output_values with tx_hash to get txId
    output_values_with_id = pd.merge(output_values, tx_hash, on='txhash', how='inner')
    
    # Merge with tx_addr to get the final mapping
    tx_addr_value = pd.merge(
        tx_addr,
        output_values_with_id[['txId', 'output_address', 'output_value']],
        on=['txId', 'output_address'],
        how='inner'
    )
    
    # Save the result
    tx_addr_value.to_csv(absolute_path('data/TxAddrValue_edgelist.csv'), index=False)
    print(f"Saved TxAddrValue_edgelist.csv with {len(tx_addr_value)} rows")

if __name__ == '__main__':
    create_value_edgelists() 