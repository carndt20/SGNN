import pandas as pd

# Check input_sent_value.csv
print("Checking input_sent_value.csv...")
df_input = pd.read_csv('data/process/input_sent_value.csv', nrows=1)
print("Columns:", df_input.columns.tolist())
print("First row:", df_input.iloc[0].to_dict())
print()

# Check AddrTx_edgelist.csv
print("Checking AddrTx_edgelist.csv...")
df_addr = pd.read_csv('data/AddrTx_edgelist.csv', nrows=1)
print("Columns:", df_addr.columns.tolist())
print("First row:", df_addr.iloc[0].to_dict())
print()

# Check if we have txId2hash.csv
print("Checking txId2hash.csv...")
df_hash = pd.read_csv('data/txId2hash.csv', nrows=1)
print("Columns:", df_hash.columns.tolist())
print("First row:", df_hash.iloc[0].to_dict()) 