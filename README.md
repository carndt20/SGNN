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


## Run

Read the [data.md](data%2Fdata.md) file before running. 
Run the following code to train the current model:
```
python train\train_address.py 
```
If you want to modify the model parameters or run other ablation models, the relevant parameters in `.config/` need to be modified.

## Testing Mode

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

### What Testing Mode Does

When enabled, testing mode:
- Processes only the specified number of transactions
- Filters all data files to maintain consistency
- Prints the number of transactions being processed
- Maintains all relationships between transactions, addresses, and features

This makes it much faster to test changes to the preprocessing pipeline while ensuring data consistency.

