import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
from elliptic_loader import EllipticLoader
from sgnn import SGNN
import warnings
import os
from numpy._core.multiarray import scalar
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import networkx as nx
import random

# Add safe globals for PyTorch 2.6
torch.serialization.add_safe_globals([scalar, np.dtype])

def calculate_class_weights(labels):
    """Calculate class weights with smoothing and maximum weight limit."""
    class_counts = torch.bincount(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    
    # Add smoothing to prevent extreme weights
    smoothing_factor = 0.1
    class_weights = total_samples / (num_classes * (class_counts + smoothing_factor))
    
    # Limit maximum weight to prevent instability
    max_weight = 5.0  # Increased from 2.0 to give more importance to minority class
    class_weights = torch.clamp(class_weights, max=max_weight)
    
    return class_weights

def find_optimal_threshold(y_true, y_pred_probs):
    """Find optimal decision threshold using precision-recall curve."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx]

def hard_negative_mining(output, labels, k=0.3):
    """Select k% of the most difficult negative samples."""
    with torch.no_grad():
        probs = torch.exp(output)
        neg_mask = (labels == 0)
        neg_probs = probs[neg_mask, 1]  # Probability of being positive class
        k = int(len(neg_probs) * k)
        _, indices = torch.topk(neg_probs, k)
        return indices

def train(model, data, optimizer, device):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    output = model(data)
    
    # Calculate class weights
    class_weights = calculate_class_weights(data.y[data.train_mask])
    class_weights = class_weights.to(device)
    
    # Get hard negative samples
    hard_neg_indices = hard_negative_mining(output[data.train_mask], data.y[data.train_mask])
    
    # Create mask for hard negatives
    hard_neg_mask = torch.zeros_like(data.train_mask)
    hard_neg_mask[data.train_mask][hard_neg_indices] = True
    
    # Combine original training mask with hard negative mask
    combined_mask = data.train_mask | hard_neg_mask
    
    # Weighted cross entropy loss
    loss = F.nll_loss(output[combined_mask], data.y[combined_mask], weight=class_weights)
    
    # Add L2 regularization
    l2_lambda = 0.01
    l2_reg = torch.tensor(0.).to(device)
    for param in model.parameters():
        l2_reg += torch.norm(param)
    loss += l2_lambda * l2_reg
    
    # Backward pass with gradient clipping
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Calculate accuracy
    pred = output[data.train_mask].argmax(dim=1)
    correct = pred.eq(data.y[data.train_mask]).sum().item()
    total = data.train_mask.sum().item()
    
    return loss.item(), correct / total

def test(model, data, device, threshold=None):
    model.eval()
    with torch.no_grad():
        output = model(data)
        
        # Get probabilities for positive class
        probs = torch.exp(output[data.test_mask])[:, 1].cpu().numpy()
        y_true = data.y[data.test_mask].cpu().numpy()
        
        # Find optimal threshold if not provided
        if threshold is None:
            threshold = find_optimal_threshold(y_true, probs)
        
        # Apply threshold
        y_pred = (probs >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Per-class metrics
        class_metrics = {}
        for class_id in [0, 1]:
            class_name = "Licit" if class_id == 0 else "Illicit"
            class_precision = precision_score(y_true, y_pred, labels=[class_id], average='binary', zero_division=0)
            class_recall = recall_score(y_true, y_pred, labels=[class_id], average='binary', zero_division=0)
            class_f1 = f1_score(y_true, y_pred, labels=[class_id], average='binary', zero_division=0)
            
            class_metrics[class_name] = {
                'precision': class_precision,
                'recall': class_recall,
                'f1': class_f1,
                'count': np.sum(y_true == class_id)
            }
        
        return accuracy, precision, recall, f1, class_metrics, cm, threshold

def visualize_graph(data, predictions, max_nodes=500):
    G = to_networkx(data.cpu(), to_undirected=True)
    # Focus on the largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        predictions = {n: predictions[n] for n in G.nodes()}
    # Sample a subgraph if too large
    if G.number_of_nodes() > max_nodes:
        sampled_nodes = random.sample(list(G.nodes()), max_nodes)
        G = G.subgraph(sampled_nodes)
        predictions = {n: predictions[n] for n in G.nodes()}
    else:
        predictions = {n: predictions[n] for n in G.nodes()}

    plt.figure(figsize=(8, 8))
    pos = nx.kamada_kawai_layout(G)
    nx.draw_networkx_edges(G, pos, alpha=0.9, width=1.2, edge_color='gray')
    color_map = ['red' if predictions[n] == 1 else 'blue' for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=30, alpha=0.8)
    plt.title("Account Graph with Model Predictions (Red: Illicit, Blue: Licit)")
    plt.axis('off')
    plt.show()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load dataset
    loader = EllipticLoader()
    data = loader.get_train_test_split()
    data = data.to(device)
    
    # Print class distribution
    train_labels = data.y[data.train_mask].cpu().numpy()
    test_labels = data.y[data.test_mask].cpu().numpy()
    print(f'Training class distribution: {np.bincount(train_labels)}')
    print(f'Testing class distribution: {np.bincount(test_labels)}')
    
    # Initialize model with enhanced architecture
    in_channels = data.num_features
    hidden_channels = 64  # Increased from 32
    out_channels = 2
    model = SGNN(in_channels, hidden_channels, out_channels, dropout=0.3, num_heads=4).to(device)
    
    # Optimizer with weight decay
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Increased learning rate
    
    # Learning rate scheduler with more aggressive reduction
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.2,  # More aggressive reduction
        patience=3,   # Reduced patience
        min_lr=1e-6,
        threshold=1e-4
    )
    
    # Training loop
    best_f1 = 0
    patience = 15  # Increased patience
    patience_counter = 0
    best_threshold = None
    
    for epoch in range(100):
        train_loss, train_acc = train(model, data, optimizer, device)
        test_acc, test_precision, test_recall, test_f1, class_metrics, cm, threshold = test(model, data, device, best_threshold)
        
        # Update learning rate
        scheduler.step(train_loss)
        
        print(f'\nEpoch: {epoch+1:03d}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Test Acc: {test_acc:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'Decision Threshold: {threshold:.4f}')
        
        # Print confusion matrix
        print('\nConfusion Matrix:')
        print(cm)
        
        # Print per-class metrics
        print('\nPer-class metrics:')
        for class_name, metrics in class_metrics.items():
            print(f'{class_name} (Count: {metrics["count"]}):')
            print(f'  Precision: {metrics["precision"]:.4f}')
            print(f'  Recall: {metrics["recall"]:.4f}')
            print(f'  F1: {metrics["f1"]:.4f}')
        
        # Early stopping
        if test_f1 > best_f1:
            best_f1 = test_f1
            best_threshold = threshold
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'threshold': threshold,
                'epoch': epoch,
                'best_f1': best_f1
            }, 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping triggered')
                break
    
    # Load best model
    checkpoint = torch.load('best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    best_threshold = checkpoint['threshold']
    
    # Final evaluation
    test_acc, test_precision, test_recall, test_f1, class_metrics, cm, _ = test(model, data, device, best_threshold)
    print(f'\nFinal Test Results:')
    print(f'Overall Accuracy: {test_acc:.4f}')
    print(f'Overall Precision: {test_precision:.4f}')
    print(f'Overall Recall: {test_recall:.4f}')
    print(f'Overall F1 Score: {test_f1:.4f}')
    print(f'Final Decision Threshold: {best_threshold:.4f}')
    
    print('\nFinal Confusion Matrix:')
    print(cm)
    
    print('\nFinal Per-class metrics:')
    for class_name, metrics in class_metrics.items():
        print(f'\n{class_name} (Count: {metrics["count"]}):')
        print(f'  Precision: {metrics["precision"]:.4f}')
        print(f'  Recall: {metrics["recall"]:.4f}')
        print(f'  F1: {metrics["f1"]:.4f}')

    # Visualize the graph with predictions
    model.eval()
    with torch.no_grad():
        output = model(data)
        predictions = output.argmax(dim=1).cpu().numpy()
    visualize_graph(data, predictions)

if __name__ == '__main__':
    main() 