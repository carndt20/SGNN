# Illicit Social Accounts? Anti-money Laundering for Transactional Blockchains

This is the source code of our paper titled "Illicit Social Accounts? Anti-money Laundering for Transactional Blockchains".
We present the PyTorch implementation of SGNN and other ablation models.


## Model
![SGNN](figures%2FModel.jpg)

## Overview
The repository is organised as follows:

- `.config/` contains the configuration files for dataset and models.
- `.data/` contains the data.
- `.dataloader/` contains the preprocessing files for the data and the data loading files for models.
- `.error/` contains the custom error class files.
- `.figures/` contains the figures of the readme.md file.
- `.models/` contains the implementation of SGNN and other ablation models.
- `.result/` contains the result of SGNN.
- `.train/` contains the training startup files of the model in the experiments.
- `.utils/` contains the tool files composed of frequently used functions.

## Environment

- `python`==3.10.10 
- `torch`==2.0.0+cu118 (CUDA 11.8)
- `pandas`==2.0.3 
- `numpy`==1.25.1
- `scikit-learn`==1.3.0

## Data

We used the Ellipse++ dataset, available at https://github.com/git-disl/EllipticPlusPlus. 
Due to space limitations, we can only provide a small portion of the data in `.data/` folder.
Please refer to the [data.md](data%2Fdata.md) file under `.data/` folder for more information about the data.

## SGNN

## How to get data
The paths to all data in the code are documented in [path.conf](config/path.conf).
We only uploaded some of the data because the files were too large to upload to github.
The .zip files need to be unzipped. 

## txId2hash.csv:
Alexander de-anonymized 99.5% transactions in 2019: https://habr.com/ru/articles/479178/ .
The data is available at [Kaggle](https://www.kaggle.com/datasets/alexbenzik/deanonymized-995-pct-of-elliptic-transactions).
Download it under `.data/` and name it txId2hash.csv.

## Files in .json/
The input_sent_value.json and output_recevied_value.json are available from [BigQuery](https://cloud.google.com/bigquery/public-data) based on txId2hash.csv.
Run [json2csv.py](dataloader/preprocess/json2csv.py) to get the input_sent_value.csv and output_recevied_value.csv.

## Other Data
If you need additional data, you can download the Elliptic++ dataset on [Github](https://github.com/git-disl/EllipticPlusPlus) to the data folder (check [path.conf](config/path.conf) for the exact path) and get the desired data by running the functions in the files of [preprocess/](dataloader/preprocess) folder and [dataset_utils.py](utils/dataset_utils.py).

## Run
Running [train_address.py](train/train_address.py) requires the 6 .npz files for [addr0_tx_mean] recorded in [path.conf](config/path.conf) and address_inOrOut_tx_time_class.csv.
The .npz files are available by running get_all_np_list() in [dataset_utils.py](utils/dataset_utils.py).

## Test Mode
For faster processing and testing, you can run the data preparation in test mode. This will process only a small subset of the data (default 1000 transactions) instead of the full dataset.

To run in test mode:
```bash
python prepare_data.py
```

The script will automatically run in test mode by default. To process the full dataset, you can modify the `test_mode` variable in `prepare_data.py`:
```python
# Test mode settings
test_mode = False  # Set to False for full data processing
test_size = 1000  # Number of transactions to process in test mode
```

## Testing Mode for process_tx.py

The preprocessing script `process_tx.py` includes a testing mode that allows you to process a smaller subset of transactions for faster testing and debugging. This is particularly useful when you want to verify the preprocessing pipeline without processing the entire dataset.

### How to Use Testing Mode

1. Open `dataloader/preprocess/process_tx.py`
2. At the bottom of the file, you'll find these settings:
```python
if __name__ == '__main__':
    # Set test_mode to True to process only a small subset of transactions
    test_mode = True
    test_size = 100  # Number of transactions to process in test mode
```

3. To enable/disable testing mode:
   - Set `test_mode = True` to process only a subset of transactions
   - Set `test_mode = False` to process the entire dataset
4. You can adjust `test_size` to control how many transactions to process in test mode


Test mode is useful for:
- Quick testing of the data processing pipeline
- Development and debugging
- Verifying the code works before running on the full dataset
- Reducing memory usage and processing time

Note: The test mode will process a subset of transactions and their related data (addresses, edges, etc.) to maintain data consistency.

## Run

Read the [data.md](data%2Fdata.md) file before running. 
Run the following code to train the current model:
```
python train\train_address.py 
```
If you want to modify the model parameters or run other ablation models, the relevant parameters in `.config/` need to be modified.

