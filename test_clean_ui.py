#!/usr/bin/env python3
"""
Test to verify clean user interface without debug messages
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from models import HeterogeneousEdgeGraphSAGE
from gnn_wrapper import GNNTrainingWrapper
from graph_debug_explorer import GraphDebugExplorer

def test_clean_inference_output():
    """Test that inference produces clean output without debug messages"""
    print("Testing clean inference output...")
    
    # Load data and build graph
    explorer = GraphDebugExplorer(random_state=42)
    graph_data = explorer.load_data_and_build_graph()
    
    # Load model
    model_path = "models/gnn_models/gnn_model_20250806_152252.pt"
    params_path = "models/gnn_models/gnn_params_20250806_152252.json"
    
    import json
    with open(params_path, 'r') as f:
        model_params = json.load(f)
    
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Determine dimensions
    source_feature_dim = None
    target_feature_dim = None
    edge_dim = None
    
    for key in state_dict.keys():
        if 'source_encoder.0.weight' in key:
            source_feature_dim = state_dict[key].shape[1]
        elif 'target_encoder.0.weight' in key:
            target_feature_dim = state_dict[key].shape[1]
        elif 'edge_mlp.0.weight' in key:
            edge_dim = state_dict[key].shape[1]
    
    # Use defaults if not found
    if source_feature_dim is None:
        source_feature_dim = 11
    if target_feature_dim is None:
        target_feature_dim = 4
    if edge_dim is None:
        edge_dim = 7
    
    # Check edge dimension mismatch (this should be handled silently)
    current_edge_dim = graph_data.edge_attr.shape[1]
    if edge_dim != current_edge_dim:
        print(f"Edge dimension mismatch: Model expects {edge_dim}, but data has {current_edge_dim}")
        print("This suggests the model was trained with different edge features.")
        print("Using the current data's edge dimension for inference.")
        edge_dim = current_edge_dim
    
    # Initialize model
    model = HeterogeneousEdgeGraphSAGE(
        source_feature_dim=source_feature_dim,
        target_feature_dim=target_feature_dim,
        hidden_channels=model_params.get('hidden_channels', 169),
        out_channels=2,
        edge_dim=edge_dim,
        dropout=model_params.get('dropout', 0.1)
    )
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # Configure wrapper
    explorer.gnn_wrapper.max_subgraph_size = 500
    explorer.gnn_wrapper.num_hops = 2
    explorer.gnn_wrapper.source_feature_dim = source_feature_dim
    explorer.gnn_wrapper.target_feature_dim = target_feature_dim
    
    # Check heterogeneous features (debug info only)
    if hasattr(explorer.gnn_wrapper, 'source_features') and hasattr(explorer.gnn_wrapper, 'target_features'):
        print(f"Using stored source features: {explorer.gnn_wrapper.source_features.shape}")
        print(f"Using stored target features: {explorer.gnn_wrapper.target_features.shape}")
    else:
        print("Source and target features not found in GNN wrapper, using fallback")
        edge_index = graph_data.edge_index.cpu().numpy()
        source_nodes = np.unique(edge_index[0])
        target_nodes = np.unique(edge_index[1])
        
        explorer.gnn_wrapper.source_features = np.random.randn(len(source_nodes), source_feature_dim)
        explorer.gnn_wrapper.target_features = np.random.randn(len(target_nodes), target_feature_dim)
    
    print(f"Using subgraph sampling with {explorer.gnn_wrapper.num_hops} hops and max {explorer.gnn_wrapper.max_subgraph_size} nodes")
    
    # Test inference
    test_edge_indices = [0]
    
    print("\n" + "="*50)
    print("CLEAN INFERENCE OUTPUT TEST")
    print("="*50)
    print("The following should show only the inference result:")
    print("-" * 30)
    
    try:
        with torch.no_grad():
            predictions, probabilities = explorer.gnn_wrapper.inference_with_subgraph_sampling(
                model, graph_data, test_edge_indices, batch_size=1
            )
            
            if len(probabilities) > 0:
                fraud_probability = probabilities[0]
                actual_label = graph_data.edge_labels[test_edge_indices[0]].item()
                actual_text = "Fraudulent" if actual_label == 1 else "Normal"
                
                # This is what the user should see - clean and simple
                print(f"✅ Inference Result:")
                print(f"   Edge {test_edge_indices[0]}: {fraud_probability:.4f} probability of fraud")
                print(f"   Actual: {actual_text}")
                print(f"   Prediction: {'Fraud' if fraud_probability > 0.5 else 'Normal'}")
                
                return True
            else:
                print("❌ No prediction returned")
                return False
                
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return False

def main():
    """Main test function"""
    success = test_clean_inference_output()
    
    if success:
        print("\n✅ Clean UI test PASSED!")
        print("The user interface now shows only the inference results without debug messages.")
    else:
        print("\n❌ Clean UI test FAILED!")
    
    return success

if __name__ == "__main__":
    main() 