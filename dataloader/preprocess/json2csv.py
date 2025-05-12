# -*- coding: utf-8 -*-

import json
import csv
import pandas as pd
from collections import defaultdict
import os
import sys
import re

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from utils.file_utils import absolute_path

def fix_malformed_json(line):
    """
    Fix malformed JSON lines where multiple addresses are concatenated.
    Example: {"txhash":"abc","output_address":"addr1","addr2","addr3","output_value":100}
    """
    # First try to parse as is
    try:
        json.loads(line)
        return [line]
    except json.JSONDecodeError:
        pass

    # Try to fix common patterns
    patterns = [
        # Pattern 1: Multiple addresses concatenated with output_value at the end
        (r'{"txhash":"([^"]+)","output_address":"([^"]+)"([^"]{25,})"output_value":(\d+)}',
         lambda m: [f'{{"txhash":"{m.group(1)}","output_address":"{addr}","output_value":{m.group(4)}}}' 
                   for addr in [m.group(2)] + re.findall(r'[13][a-km-zA-HJ-NP-Z1-9]{25,34}', m.group(3))]),
        
        # Pattern 2: Multiple addresses concatenated without quotes
        (r'{"txhash":"([^"]+)","output_address":"([^"]+)",([^"]{25,34}),([^"]{25,34})"output_value":(\d+)}',
         lambda m: [f'{{"txhash":"{m.group(1)}","output_address":"{addr}","output_value":{m.group(5)}}}' 
                   for addr in [m.group(2), m.group(3), m.group(4)]])
    ]

    for pattern, fix_func in patterns:
        match = re.search(pattern, line)
        if match:
            try:
                fixed_lines = fix_func(match)
                # Validate each fixed line
                for fixed_line in fixed_lines:
                    json.loads(fixed_line)
                return fixed_lines
            except (json.JSONDecodeError, IndexError):
                continue

    # If no patterns match, try to extract addresses and create valid JSON
    try:
        # Extract txhash
        txhash_match = re.search(r'"txhash":"([^"]+)"', line)
        if not txhash_match:
            return [line]
        txhash = txhash_match.group(1)

        # Extract output_value
        value_match = re.search(r'"output_value":(\d+)}', line)
        if not value_match:
            return [line]
        value = value_match.group(1)

        # Extract all Bitcoin addresses
        addresses = re.findall(r'[13][a-km-zA-HJ-NP-Z1-9]{25,34}', line)
        if not addresses:
            return [line]

        # Create valid JSON for each address
        return [f'{{"txhash":"{txhash}","output_address":"{addr}","output_value":{value}}}' 
                for addr in addresses]
    except Exception:
        return [line]

def json2csv(json_path, csv_path, is_input=True):
    """
    Convert data from json to csv file, properly handling multiple addresses per transaction
    
    Args:
        json_path: Path to input JSON file
        csv_path: Path to output CSV file
        is_input: True if processing input addresses, False if processing output addresses
    """
    # Read the JSON file line by line
    transactions = defaultdict(list)
    line_number = 0
    error_count = 0
    fixed_count = 0
    skipped_count = 0
    
    print(f"Reading {json_path}...")
    with open(json_path, 'r') as f:
        for line in f:
            line_number += 1
            try:
                # Try to fix malformed JSON lines
                fixed_lines = fix_malformed_json(line)
                if len(fixed_lines) > 1:
                    fixed_count += 1
                
                for fixed_line in fixed_lines:
                    try:
                        data = json.loads(fixed_line)
                        txhash = data['txhash']
                        address_key = 'input_address' if is_input else 'output_address'
                        value_key = 'input_value' if is_input else 'output_value'
                        
                        if address_key not in data or value_key not in data:
                            skipped_count += 1
                            continue
                            
                        transactions[txhash].append({
                            address_key: data[address_key],
                            value_key: data[value_key]
                        })
                    except json.JSONDecodeError as e:
                        error_count += 1
                        if error_count <= 5:  # Only show first 5 errors
                            print(f"Error on line {line_number}: {str(e)}")
                            print(f"Problematic line: {fixed_line[:200]}...")  # Show first 200 chars
                        continue
                    except KeyError as e:
                        error_count += 1
                        if error_count <= 5:  # Only show first 5 errors
                            print(f"Missing key on line {line_number}: {str(e)}")
                            print(f"Available keys: {list(data.keys())}")
                        continue
            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    print(f"Unexpected error on line {line_number}: {str(e)}")
                continue
    
    if error_count > 0:
        print(f"\nEncountered {error_count} errors while processing the file")
        if error_count > 5:
            print("(Only showing first 5 errors)")
    if fixed_count > 0:
        print(f"Fixed {fixed_count} malformed JSON lines")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} lines with missing required fields")
    
    # Convert to DataFrame format
    rows = []
    for txhash, addresses in transactions.items():
        # If there's only one address, keep it as is
        if len(addresses) == 1:
            rows.append({
                'txhash': txhash,
                **addresses[0]
            })
        else:
            # For multiple addresses, create a row for each
            for addr_data in addresses:
                rows.append({
                    'txhash': txhash,
                    **addr_data
                })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(rows)
    print(f"Found {len(transactions)} unique transactions")
    print(f"Total addresses: {len(df)}")
    print("\nSample of transactions with multiple addresses:")
    multi_addr_txs = {tx: addrs for tx, addrs in transactions.items() if len(addrs) > 1}
    for txhash, addrs in list(multi_addr_txs.items())[:5]:
        print(f"\nTransaction: {txhash}")
        for addr_data in addrs:
            print(f"  Address: {addr_data[address_key]}, Value: {addr_data[value_key]}")
    
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")

def get_received_value():
    df = pd.read_csv('../../data/process/output_recevied_value.csv')
    # No need to replace brackets anymore as we're handling the data properly
    df.to_csv('../../data/process/output_recevied_value.csv', index=False)

if __name__ == '__main__':
    # Process input addresses
    json2csv(
        absolute_path('data/json/input_sent_value.json'),
        absolute_path('data/process/input_sent_value.csv'),
        is_input=True
    )
    
    # Process output addresses
    json2csv(
        absolute_path('data/json/output_recevied_value.json'),
        absolute_path('data/process/output_recevied_value.csv'),
        is_input=False
    )
    get_received_value()