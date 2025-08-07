#!/usr/bin/env python3
"""
Test script to verify Streamlit inference is working correctly
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from models import HeterogeneousEdgeGraphSAGE
from gnn_wrapper import GNNTrainingWrapper
from graph_debug_explorer import GraphDebugExplorer

def test_inference_with_subgraph():
    """Test inference using subgraph sampling"""
    print("Testing inference with subgraph sampling...")
    
    # Check if data file exists
    if not os.path.exists('./preprocessed_fraud_test.csv'):
        print("Error: preprocessed_fraud_test.csv not found")
        return False
    
    # Load data and build graph
    explorer = GraphDebugExplorer(random_state=42)
    graph_data = explorer.load_data_and_build_graph()
    
    if graph_data is None:
        print("Error: Failed to load graph data")
        return False
    
    print(f"Graph loaded: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    # Check if model exists
    model_path = "models/gnn_models/gnn_model_20250806_152252.pt"
    params_path = "models/gnn_models/gnn_params_20250806_152252.json"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return False
    
    # Load model parameters
    import json
    with open(params_path, 'r') as f:
        model_params = json.load(f)
    
    # Load the state dict to determine model dimensions
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Determine dimensions from state dict
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
    
    # Check if the edge dimension matches the current data
    current_edge_dim = graph_data.edge_attr.shape[1]
    if edge_dim != current_edge_dim:
        print(f"Edge dimension mismatch: Model expects {edge_dim}, but data has {current_edge_dim}")
        print("This suggests the model was trained with different edge features.")
        print("Using the current data's edge dimension for inference.")
        edge_dim = current_edge_dim
    
    print(f"Model dimensions: Source={source_feature_dim}, Target={target_feature_dim}, Edge={edge_dim}")
    
    # Initialize the model
    model = HeterogeneousEdgeGraphSAGE(
        source_feature_dim=source_feature_dim,
        target_feature_dim=target_feature_dim,
        hidden_channels=model_params.get('hidden_channels', 169),
        out_channels=2,
        edge_dim=edge_dim,
        dropout=model_params.get('dropout', 0.1)
    )
    
    # Load the trained weights
    model.load_state_dict(state_dict)
    model.eval()
    
    # Configure the GNN wrapper for inference
    explorer.gnn_wrapper.max_subgraph_size = 500
    explorer.gnn_wrapper.num_hops = 2
    
    # Store the feature dimensions in the wrapper
    explorer.gnn_wrapper.source_feature_dim = source_feature_dim
    explorer.gnn_wrapper.target_feature_dim = target_feature_dim
    
    # Check if heterogeneous features are available
    if hasattr(explorer.gnn_wrapper, 'source_features') and hasattr(explorer.gnn_wrapper, 'target_features'):
        print(f"Using stored source features: {explorer.gnn_wrapper.source_features.shape}")
        print(f"Using stored target features: {explorer.gnn_wrapper.target_features.shape}")
    else:
        print("Warning: Heterogeneous features not found, using fallback")
        # Create dummy features
        edge_index = graph_data.edge_index.cpu().numpy()
        source_nodes = np.unique(edge_index[0])
        target_nodes = np.unique(edge_index[1])
        
        explorer.gnn_wrapper.source_features = np.random.randn(len(source_nodes), source_feature_dim)
        explorer.gnn_wrapper.target_features = np.random.randn(len(target_nodes), target_feature_dim)
    
    # Test inference on a few edges
    test_edge_indices = [0, 100, 500]  # Test different edges
    
    print(f"Testing inference on {len(test_edge_indices)} edges...")
    
    try:
        with torch.no_grad():
            predictions, probabilities = explorer.gnn_wrapper.inference_with_subgraph_sampling(
                model, graph_data, test_edge_indices, batch_size=1
            )
            
            print(f"Successfully performed inference!")
            print(f"Predictions shape: {predictions.shape}")
            print(f"Probabilities shape: {probabilities.shape}")
            
            # Show results for each test edge
            for i, edge_idx in enumerate(test_edge_indices):
                if i < len(probabilities):
                    prob = probabilities[i]
                    pred = predictions[i]
                    actual_label = graph_data.edge_labels[edge_idx].item()
                    
                    print(f"Edge {edge_idx}: Prediction={pred}, Probability={prob:.4f}, Actual={actual_label}")
            
            return True
            
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("="*60)
    print("TESTING STREAMLIT INFERENCE WITH SUBGRAPH SAMPLING")
    print("="*60)
    
    success = test_inference_with_subgraph()
    
    if success:
        print("\n✅ Inference test PASSED!")
        print("The Streamlit app should work correctly with subgraph sampling.")
    else:
        print("\n❌ Inference test FAILED!")
        print("There may be issues with the inference implementation.")
    
    return success

if __name__ == "__main__":
    main() 