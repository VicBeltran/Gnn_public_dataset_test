#!/usr/bin/env python3
"""
Test script to verify that both XGBoost and GNN models evaluate on the same dataset size
"""

import pandas as pd
import numpy as np
from evaluate_models import ModelEvaluator

def test_evaluation_consistency():
    """Test that both models evaluate on the same dataset size"""
    print("Testing evaluation consistency...")
    
    # Create evaluator
    evaluator = ModelEvaluator(random_state=42)
    
    # Load and prepare data
    df, node_features, edge_features, source_node_features, target_node_features = evaluator.load_and_prepare_data()
    
    print(f"\nDataset size: {len(df)} samples")
    
    # Test XGBoost data preparation
    X_xgb, y_xgb = evaluator.prepare_tabular_data(
        df, node_features + edge_features, 'is_fraud'
    )
    print(f"XGBoost evaluation data size: {len(X_xgb)} samples")
    
    # Test GNN data preparation
    full_data = evaluator.prepare_graph_data(
        df, 'cc_num', 'merchant', source_node_features, target_node_features, 
        edge_features, 'is_fraud'
    )
    print(f"GNN evaluation data size: {full_data.num_edges} edges")
    
    # Verify consistency
    if len(X_xgb) == full_data.num_edges:
        print("✅ SUCCESS: Both models will evaluate on the same dataset size!")
    else:
        print("❌ ERROR: Models will evaluate on different dataset sizes!")
        print(f"  XGBoost: {len(X_xgb)} samples")
        print(f"  GNN: {full_data.num_edges} edges")
    
    return len(X_xgb) == full_data.num_edges

if __name__ == "__main__":
    success = test_evaluation_consistency()
    if success:
        print("\n🎉 The fix is working correctly!")
    else:
        print("\n💥 There's still an issue with the evaluation consistency.") 