# -*- coding: utf-8 -*-

import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import pandas as pd
import numpy as np
from utils.file_utils import absolute_path
from utils.config_utils import get_config


def get_tx_by_address(test_mode=False, test_size=100):
    config = get_config('path')
    print("Reading transaction classes...")
    tx_class_df = pd.read_csv(absolute_path(config['file']['txs_classes']))
    
    if test_mode:
        print(f"Test mode: Using {test_size} transactions")
        # Get a small subset of transactions that have both input and output addresses
        print("Reading address transaction mappings...")
        addrTx_df = pd.read_csv(absolute_path(config['file']['addrTx']))
        txAddr_df = pd.read_csv(absolute_path(config['file']['txAddr']))
        
        # Get valid transactions first
        valid_txs = tx_class_df[tx_class_df.apply(lambda x: x['txId'] in addrTx_df['txId'].values
                                                          and x['txId'] in txAddr_df['txId'].values, axis=1)]
        tx_class_df = valid_txs.head(test_size)
        print(f"Selected {len(tx_class_df)} transactions for testing")
    
    tx_class_df.to_csv(absolute_path(config['file']['txs_classes_new']), index=False)


def get_tx_edge_feature_by_tx(test_mode=False, test_size=100):
    config = get_config('path')
    print("Reading transaction classes...")
    tx_class_df = pd.read_csv(absolute_path(config['file']['txs_classes_new']))
    test_txs = set(tx_class_df['txId'])
    
    print("Reading transaction features...")
    tx_feature_df = pd.read_csv(absolute_path(config['file']['txs_features']))
    tx_feature_df = tx_feature_df[tx_feature_df['txId'].isin(test_txs)]
    print(f"Features shape: {tx_feature_df.shape}")
    
    print("Reading transaction edges...")
    tx_edge_df = pd.read_csv(absolute_path(config['file']['txs_edges']))
    tx_edge_df = tx_edge_df[tx_edge_df['txId1'].isin(test_txs) & tx_edge_df['txId2'].isin(test_txs)]
    print(f"Edges shape: {tx_edge_df.shape}")
    
    print("Reading transaction hashes...")
    txId2hash = pd.read_csv(absolute_path(config['file']['txId2hash']))
    txId2hash = txId2hash[txId2hash['txId'].isin(test_txs)]
    print(f"Hash mapping shape: {txId2hash.shape}")
    
    print("Reading transaction timestamps...")
    tx_time_df = pd.read_csv(absolute_path(config['file']['txs_time']))
    print(f"Time data shape: {tx_time_df.shape}")
    
    # Merge hashes with timestamps
    print("Merging transaction data...")
    # First merge txId2hash with tx_time_df
    tx_time_df = pd.merge(txId2hash, tx_time_df, how='left', on='txhash')
    print(f"After first merge shape: {tx_time_df.shape}")
    print("Columns in tx_time_df:", tx_time_df.columns.tolist())
    
    # Then merge with features
    tx_feature_df = pd.merge(tx_time_df, tx_feature_df, on='txId')
    print(f"After second merge shape: {tx_feature_df.shape}")
    
    assert tx_feature_df.shape[0] == tx_time_df.shape[0], 'Some txs do not have timestamp!'
    
    print("Saving processed files...")
    tx_time_df.to_csv(absolute_path(config['file']['txs_time_new']), index=False)
    tx_feature_df.to_csv(absolute_path(config['file']['txs_features_new']), index=False)
    tx_edge_df.to_csv(absolute_path(config['file']['txs_edges_new']), index=False)


def tx_indegree(test_mode=False, test_size=100):
    """input: addrTx or txAddr"""
    config = get_config('path')
    print("Reading transaction classes...")
    test_txs = pd.read_csv(absolute_path(config['file']['txs_classes_new']))['txId']
    
    print("Reading address transaction mappings...")
    addrTx = pd.read_csv(absolute_path(config['file']['addrTx']))
    txAddr = pd.read_csv(absolute_path(config['file']['txAddr']))
    
    # Filter to test transactions
    addrTx = addrTx[addrTx['txId'].isin(test_txs)]
    txAddr = txAddr[txAddr['txId'].isin(test_txs)]
    
    print("Calculating degrees...")
    indegree = addrTx.groupby(['txId']).size().reset_index()
    indegree.columns = ['txId', 'indegree']
    indegree.to_csv(absolute_path(config['file']['txs_indegree']), index=False)
    
    outdegree = txAddr.groupby(['txId']).size().reset_index()
    outdegree.columns = ['txId', 'outdegree']
    outdegree.to_csv(absolute_path(config['file']['txs_outdegree']), index=False)
    
    degree = pd.merge(indegree, outdegree, on='txId')
    degree.to_csv(absolute_path(config['file']['txs_degree']), index=False)


def tx_total_values(test_mode=False, test_size=100):
    config = get_config('path')
    print("Reading transaction classes...")
    test_txs = pd.read_csv(absolute_path(config['file']['txs_classes_new']))['txId']
    
    print("Reading transaction values...")
    addrTxValue = pd.read_csv(absolute_path(config['file']['addrTxValue']))
    txAddrValue = pd.read_csv(absolute_path(config['file']['txAddrValue']))
    
    # Filter to test transactions
    addrTxValue = addrTxValue[addrTxValue['txId'].isin(test_txs)]
    txAddrValue = txAddrValue[txAddrValue['txId'].isin(test_txs)]
    
    print("Calculating total values...")
    in_total_values = addrTxValue.groupby(['txId'])['input_value'].sum().reset_index()
    in_total_values.columns = ['txId', 'in_total_values']
    in_total_values.to_csv(absolute_path(config['file']['input_total_values_new']), index=False)
    
    out_total_values = txAddrValue.groupby(['txId'])['output_value'].sum().reset_index()
    out_total_values.columns = ['txId', 'out_total_values']
    out_total_values.to_csv(absolute_path(config['file']['output_total_values_new']), index=False)
    
    total_values = pd.merge(in_total_values, out_total_values, on='txId')
    total_values['values_difference'] = total_values['in_total_values'] - total_values['out_total_values']
    total_values.to_csv(absolute_path(config['file']['txs_total_values_new']), index=False)


def txs_some_feature(test_mode=False, test_size=100):
    config = get_config('path')
    print("Reading transaction classes...")
    txs_classes = pd.read_csv(absolute_path(config['file']['txs_classes_new']))
    test_txs = set(txs_classes['txId'])
    
    print("Reading transaction degrees...")
    txs_degree = pd.read_csv(absolute_path(config['file']['txs_degree']))
    txs_degree = txs_degree[txs_degree['txId'].isin(test_txs)]
    
    print("Reading transaction values...")
    txs_total_values = pd.read_csv(absolute_path(config['file']['txs_total_values_new']))
    txs_total_values = txs_total_values[txs_total_values['txId'].isin(test_txs)]
    
    print("Combining features...")
    txs_some_feature = pd.merge(txs_degree, txs_total_values, on='txId')
    txs_some_feature = pd.merge(txs_some_feature, txs_classes, on='txId')
    assert txs_some_feature.shape[0] == txs_classes.shape[0]
    txs_some_feature.to_csv(absolute_path(config['file']['txs_some_feature_new']), index=False)


if __name__ == '__main__':
    # Set test_mode to True to process only a small subset of transactions
    test_mode = True
    test_size = 100  # Number of transactions to process in test mode
    
    print("Step 1: Processing transaction classes...")
    get_tx_by_address(test_mode, test_size)
    
    print("\nStep 2: Processing transaction edge features...")
    get_tx_edge_feature_by_tx(test_mode, test_size)
    
    print("\nStep 3: Calculating transaction degrees...")
    tx_indegree(test_mode, test_size)
    
    print("\nStep 4: Calculating transaction values...")
    tx_total_values(test_mode, test_size)
    
    print("\nStep 5: Combining transaction features...")
    txs_some_feature(test_mode, test_size)