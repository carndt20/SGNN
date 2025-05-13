# SGNN: Subgraph Neural Network for Graph Classification

## Overview
This project implements a Subgraph Neural Network (SGNN) for graph-based tasks, specifically designed for classifying nodes in a graph (e.g., licit vs. illicit accounts in a transaction network). The model uses a combination of Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT) to capture both structural and transactional features.

## Project Structure
- **sgnn.py**: Contains the core SGNN model implementation, including the MultiHeadAttention mechanism and the main SGNN class.
- **train.py**: Script for training and evaluating the model, including data loading, training loop, and visualization of the graph.
- **elliptic_loader.py**: Data loader for the Elliptic dataset, handling feature extraction, edge loading, and label processing.
- **requirements.txt**: Lists the Python dependencies required to run the project.
- **best_model.pt**: Saved PyTorch model checkpoint of the best-performing model.

## Requirements
- Python 3.6–3.10 recommended (Python 3.12+ may have compatibility issues with some dependencies)
- PyTorch
- PyTorch Geometric
- NetworkX
- Matplotlib
- NumPy
- scikit-learn

It's recommended to set up a Python virtual environment before installing dependencies:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
1. **Data Preparation**: 
   - The dataset used in this project is available from the [EllipticPlusPlus repository](https://github.com/git-disl/EllipticPlusPlus).
   - Ensure your dataset is in the correct format and directory. By default, the code expects the data files (e.g., `txs_features.csv`, `txs_edgelist.csv`, `txs_classes.csv`) to be in a folder named `data` in the project root. You can change this in `elliptic_loader.py` if your data is elsewhere.
2. **Training**: Run the training script:
   ```bash
   python train.py
   ```
   This will train the model, evaluate its performance, and visualize the graph with node predictions.

3. **Visualization**: The training script includes a visualization function that plots the graph, with nodes colored by their predicted class (red for illicit, blue for licit). The visualization is limited to the largest connected component and a maximum of 500 nodes for clarity.

## Model Architecture
- **Transaction Network**: Uses Graph Attention Network (GAT) layers to capture transactional features between nodes.
- **Identity Network**: Uses Graph Convolutional Network (GCN) layers to capture structural (identity-based) features of nodes.
- **Feature Fusion**: Combines features from both networks using a fusion layer and a Multi-Head Self-Attention mechanism.
- **Classification**: The final classification layer outputs node predictions (e.g., licit or illicit).

## Performance Metrics
The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Troubleshooting
### 1. PyTorch Geometric Warnings (e.g., 'torch-scatter', 'torch-sparse')
- **Solution:**
  - Make sure you are using a compatible Python and PyTorch version (Python 3.6–3.10 is safest).
  - Install PyTorch Geometric and its dependencies following the official guide: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
  - For CPU-only install (example for PyTorch 2.7.0):
    ```bash
    pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.7.0+cpu.html
    ```
  - If you use a different PyTorch version, change the URL accordingly.

### 2. General Advice
- If you encounter other errors, check the error message for missing files or modules and ensure all dependencies are installed and data paths are correct.
- If you use a different dataset or directory structure, update the paths in `elliptic_loader.py` accordingly.

