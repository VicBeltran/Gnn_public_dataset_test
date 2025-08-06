#!/usr/bin/env python3
"""
Test script to verify GNN training fixes
"""

import numpy as np
import pandas as pd
import torch
from gnn_wrapper import GNNTrainingWrapper
from training_pipeline import FocalLoss

def test_focal_loss():
    """Test the improved focal loss implementation"""
    print("Testing Focal Loss...")
    
    # Create test data
    batch_size = 100
    num_classes = 2
    
    # Create logits and targets
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test focal loss
    focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
    loss = focal_loss(logits, targets)
    
    print(f"Focal loss value: {loss.item():.6f}")
    print(f"Loss is finite: {torch.isfinite(loss).item()}")
    print(f"Loss is positive: {(loss > 0).item()}")
    
    return loss.item()

def test_gnn_training():
    """Test GNN training with the fixes"""
    print("\nTesting GNN Training...")
    
    # Create synthetic data
    np.random.seed(42)
    n_interactions = 500
    n_nodes = 50
    
    # Create synthetic interaction data
    df = pd.DataFrame({
        'source_node': np.random.randint(0, n_nodes, n_interactions),
        'target_node': np.random.randint(0, n_nodes, n_interactions),
        'amount': np.random.exponential(100, n_interactions),
        'hour': np.random.randint(0, 24, n_interactions),
        'day_of_week': np.random.randint(0, 7, n_interactions),
        'is_fraud': np.random.choice([0, 1], n_interactions, p=[0.9, 0.1])  # 10% fraud rate
    })
    
    # Define features
    node_features = ['amount', 'hour', 'day_of_week']
    edge_features = ['amount', 'hour', 'day_of_week']
    
    # Create wrapper
    wrapper = GNNTrainingWrapper(random_state=42, max_subgraph_size=100)
    
    # Build graph
    data = wrapper.build_graph_from_tabular(
        df=df,
        node_features=node_features,
        edge_features=edge_features,
        target_col='is_fraud',
        source_node_col='source_node',
        target_node_col='target_node'
    )
    
    # Split data
    train_data, val_data, test_data = wrapper.split_graph_data_with_sampling(data)
    
    print(f"Train data: {train_data.num_nodes} nodes, {train_data.num_edges} edges")
    print(f"Val data: {val_data.num_nodes} nodes, {val_data.num_edges} edges")
    print(f"Test data: {test_data.num_nodes} nodes, {test_data.num_edges} edges")
    
    # Test optimization with fewer trials
    print("\nTesting optimization...")
    model, results, best_params = wrapper.train_gnn_with_optimization(
        train_data, val_data, test_data,
        n_trials=3,  # Just a few trials for testing
        final_training=True
    )
    
    if results is not None:
        print(f"Test AUC: {results['auc']:.4f}")
        print(f"Test Accuracy: {results['accuracy']:.4f}")
        print(f"Test F1-Score: {results['f1_score']:.4f}")
    
    return model, results, best_params

def test_heterogeneous_training():
    """Test heterogeneous GNN training"""
    print("\nTesting Heterogeneous GNN Training...")
    
    # Create synthetic data with different feature sets
    np.random.seed(42)
    n_transactions = 1000
    n_customers = 50
    n_merchants = 25
    
    # Create synthetic transaction data with different feature sets
    df = pd.DataFrame({
        # Node identifiers
        'customer_id': np.random.randint(0, n_customers, n_transactions),
        'merchant_id': np.random.randint(0, n_merchants, n_transactions),
        
        # Source node features (customer-specific) - 4 features
        'customer_age': np.random.randint(18, 80, n_transactions),
        'customer_credit_score': np.random.randint(300, 850, n_transactions),
        'customer_account_balance': np.random.exponential(5000, n_transactions),
        'customer_risk_score': np.random.uniform(0, 1, n_transactions),
        
        # Target node features (merchant-specific) - 3 features
        'merchant_category': np.random.choice(['retail', 'food', 'travel', 'electronics', 'services'], n_transactions),
        'merchant_risk_score': np.random.uniform(0, 1, n_transactions),
        'merchant_transaction_volume': np.random.exponential(10000, n_transactions),
        
        # Edge features (transaction-specific)
        'amount': np.random.exponential(100, n_transactions),
        'hour': np.random.randint(0, 24, n_transactions),
        'day_of_week': np.random.randint(0, 7, n_transactions),
        
        # Target variable
        'is_fraud': np.random.choice([0, 1], n_transactions, p=[0.9, 0.1])
    })
    
    # Encode merchant category
    df['merchant_category_encoded'] = df['merchant_category'].map({
        'retail': 0, 'food': 1, 'travel': 2, 'electronics': 3, 'services': 4
    })
    
    # Define different feature sets
    source_node_features = [
        'customer_age', 'customer_credit_score', 'customer_account_balance', 
        'customer_risk_score'
    ]  # 4 features
    
    target_node_features = [
        'merchant_category_encoded', 'merchant_risk_score', 'merchant_transaction_volume'
    ]  # 3 features
    
    edge_features = ['amount', 'hour', 'day_of_week']
    
    print(f"Source node features ({len(source_node_features)}): {source_node_features}")
    print(f"Target node features ({len(target_node_features)}): {target_node_features}")
    print(f"Edge features ({len(edge_features)}): {edge_features}")
    
    # Create wrapper
    wrapper = GNNTrainingWrapper(random_state=42, max_subgraph_size=100)
    
    # Build graph using heterogeneous approach
    data = wrapper.build_graph_from_tabular(
        df=df,
        node_features=source_node_features + target_node_features,  # Combined for compatibility
        edge_features=edge_features,
        target_col='is_fraud',
        source_node_col='customer_id',
        target_node_col='merchant_id',
        source_node_features=source_node_features,  # Different feature sets
        target_node_features=target_node_features
    )
    
    # Split data
    train_data, val_data, test_data = wrapper.split_graph_data_with_sampling(data)
    
    print(f"Train data: {train_data.num_nodes} nodes, {train_data.num_edges} edges")
    print(f"Val data: {val_data.num_nodes} nodes, {val_data.num_edges} edges")
    print(f"Test data: {test_data.num_nodes} nodes, {test_data.num_edges} edges")
    print(f"Source feature dimension: {wrapper.source_feature_dim}")
    print(f"Target feature dimension: {wrapper.target_feature_dim}")
    
    # Test optimization with fewer trials
    print("\nTesting heterogeneous optimization...")
    model, results, best_params = wrapper.train_gnn_with_optimization(
        train_data, val_data, test_data,
        n_trials=2,  # Just a few trials for testing
        final_training=True
    )
    
    if results is not None:
        print(f"Test AUC: {results['auc']:.4f}")
        print(f"Test Accuracy: {results['accuracy']:.4f}")
        print(f"Test F1-Score: {results['f1_score']:.4f}")
    
    return model, results, best_params

if __name__ == "__main__":
    print("Testing GNN Training Fixes")
    print("=" * 50)
    
    # Test focal loss
    test_focal_loss()
    
    # Test standard GNN training
    test_gnn_training()
    
    # Test heterogeneous GNN training
    test_heterogeneous_training()
    
    print("\nAll tests completed!") 