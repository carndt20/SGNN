# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import psutil

from utils.config_utils import get_config
from utils.file_utils import absolute_path

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def get_address_feature_no_repeat():
    config = get_config('path')
    df = pd.read_csv(absolute_path(config['file']['addr_features']))
    df.drop_duplicates().to_csv(absolute_path(config['file']['addr_features_no_repeat']), index=False)

def process_in_chunks(file_path, chunk_size=50000):
    """Process a CSV file in chunks to avoid memory issues."""
    print(f"Reading {file_path} in chunks of {chunk_size} rows...")
    chunks = pd.read_csv(absolute_path(file_path), chunksize=chunk_size)
    return chunks

def get_address_feature_with_tx_feature_repeat():
    print("Processing address features with transaction features...")
    print(f"Current memory usage: {get_memory_usage():.2f} GB")
    
    config = get_config('path')
    
    # Read address features
    print("Reading address features...")
    addr_features = pd.read_csv(absolute_path(config['file']['addr_features']))
    addr_features = addr_features.drop(columns=['Time step']).drop_duplicates()
    print(f"Address features shape: {addr_features.shape}")
    print(f"Memory usage after reading address features: {get_memory_usage():.2f} GB")
    
    # Read address-transaction mappings
    print("Reading address-transaction mappings...")
    addrTx = pd.read_csv(absolute_path(config['file']['addrTx']))
    txAddr = pd.read_csv(absolute_path(config['file']['txAddr']))
    addrTx.columns = ['address', 'txId']
    txAddr.columns = ['txId', 'address']
    
    # Combine address-transaction mappings
    print("Combining address-transaction mappings...")
    addr_tx = pd.concat([addrTx[['address', 'txId']], txAddr[['address', 'txId']]], axis=0)
    assert addr_tx.shape[0] == addrTx.shape[0] + txAddr.shape[0]
    print(f"Combined address-transaction mappings shape: {addr_tx.shape}")
    
    # Merge with address features
    print("Merging with address features...")
    addr_features_all = pd.merge(addr_tx, addr_features, on='address')
    assert addr_features_all.shape[0] == addr_tx.shape[0]
    print(f"Merged features shape: {addr_features_all.shape}")
    print(f"Memory usage after merging: {get_memory_usage():.2f} GB")
    
    # Process transaction features in chunks
    print("Processing transaction features in chunks...")
    output_chunks = []
    tx_features_chunks = process_in_chunks(config['file']['txs_features'])
    
    for chunk in tqdm(tx_features_chunks, desc="Processing chunks"):
        merged_chunk = pd.merge(addr_features_all, chunk, on='txId')
        if not merged_chunk.empty:
            output_chunks.append(merged_chunk)
        if len(output_chunks) * 50000 > 1000000:  # If we have more than 1M rows in memory
            # Combine and save intermediate results
            print("Saving intermediate results...")
            intermediate = pd.concat(output_chunks, ignore_index=True)
            intermediate.to_csv(absolute_path(config['file']['addr_features_with_tx_features']), 
                              mode='a', header=not os.path.exists(absolute_path(config['file']['addr_features_with_tx_features'])), 
                              index=False)
            output_chunks = []
            print(f"Memory usage after saving intermediate: {get_memory_usage():.2f} GB")
    
    # Combine remaining chunks
    if output_chunks:
        print("Combining remaining chunks...")
        addr_features_with_tx_features = pd.concat(output_chunks, ignore_index=True)
        addr_features_with_tx_features.to_csv(absolute_path(config['file']['addr_features_with_tx_features']), 
                                            mode='a', header=not os.path.exists(absolute_path(config['file']['addr_features_with_tx_features'])), 
                                            index=False)
    
    print("Done!")
    print(f"Final memory usage: {get_memory_usage():.2f} GB")

def get_addr_feature_with_tx_feature_mean():
    print("Calculating mean features...")
    print(f"Current memory usage: {get_memory_usage():.2f} GB")
    
    config = get_config('path')
    
    # Process in chunks
    chunks = process_in_chunks(config['file']['addr_features_with_tx_features'])
    mean_chunks = []
    
    for chunk in tqdm(chunks, desc="Calculating means"):
        chunk = chunk.drop('txId', axis=1)
        mean_chunk = chunk.groupby(['address', 'Time step']).mean().reset_index()
        mean_chunks.append(mean_chunk)
        if len(mean_chunks) * 50000 > 1000000:  # If we have more than 1M rows in memory
            # Combine and save intermediate results
            print("Saving intermediate means...")
            intermediate = pd.concat(mean_chunks, ignore_index=True)
            intermediate = intermediate.groupby(['address', 'Time step']).mean().reset_index()
            intermediate.to_csv(absolute_path(config['file']['addr_feature_with_tx_feature_mean']), 
                              mode='a', header=not os.path.exists(absolute_path(config['file']['addr_feature_with_tx_feature_mean'])), 
                              index=False)
            mean_chunks = []
            print(f"Memory usage after saving intermediate: {get_memory_usage():.2f} GB")
    
    # Combine remaining chunks
    if mean_chunks:
        print("Combining remaining means...")
        addr_feature_with_tx_feature_mean = pd.concat(mean_chunks, ignore_index=True)
        addr_feature_with_tx_feature_mean = addr_feature_with_tx_feature_mean.groupby(['address', 'Time step']).mean().reset_index()
        addr_feature_with_tx_feature_mean.to_csv(absolute_path(config['file']['addr_feature_with_tx_feature_mean']), 
                                               mode='a', header=not os.path.exists(absolute_path(config['file']['addr_feature_with_tx_feature_mean'])), 
                                               index=False)
    
    print("Done!")
    print(f"Final memory usage: {get_memory_usage():.2f} GB")

def get_address_edge_new():
    print("Processing address edges...")
    print(f"Current memory usage: {get_memory_usage():.2f} GB")
    
    config = get_config('path')
    
    # Read files
    print("Reading edge files...")
    addrTx = pd.read_csv(absolute_path(config['file']['addrTx']))
    txAddr = pd.read_csv(absolute_path(config['file']['txAddr']))
    addr_edges = pd.read_csv(absolute_path(config['file']['addr_edges']))
    
    # Process in chunks
    print("Merging edge data...")
    addr_edges_new = pd.merge(addrTx, txAddr, on='txId')
    assert addr_edges.shape[0] == addr_edges_new.shape[0], f'merge error！addr_edges:{addr_edges.shape[0]}, addr_edges_new:{addr_edges_new.shape[0]}'
    
    # Save to CSV
    print("Saving results...")
    addr_edges_new.to_csv(absolute_path(config['file']['addr_edges_new']), index=False)
    print(f'addr_edges:{addr_edges.shape[0]}, addr_edges_new:{addr_edges_new.shape[0]}')
    print("Done!")
    print(f"Final memory usage: {get_memory_usage():.2f} GB")

if __name__ == '__main__':
    # First get address features with transaction features
    get_address_feature_with_tx_feature_repeat()
    # Then calculate mean features
    get_addr_feature_with_tx_feature_mean()
    # Finally process edges
    get_address_edge_new()

