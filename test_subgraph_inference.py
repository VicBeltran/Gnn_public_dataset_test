"""
Test script to demonstrate subgraph inference evaluation
This shows how the GNN model now uses subgraph sampling for realistic inference.
"""

from evaluate_models import ModelEvaluator
from graph_debug_explorer import GraphDebugExplorer

def test_subgraph_inference():
    """Test the subgraph inference evaluation"""
    print("Testing Subgraph Inference Evaluation")
    print("="*50)
    
    # Create evaluator
    evaluator = ModelEvaluator(random_state=42)
    
    # Load data and models
    print("Loading data and models...")
    df, node_features, edge_features, source_node_features, target_node_features = evaluator.load_and_prepare_data()
    evaluator.load_models()
    
    # Prepare graph data
    print("Preparing graph data...")
    full_data = evaluator.prepare_graph_data(
        df, 'cc_num', 'merchant', source_node_features, target_node_features, 
        edge_features, 'is_fraud'
    )
    
    # Test GNN evaluation with subgraph sampling
    print("\nTesting GNN evaluation with subgraph sampling...")
    gnn_results = evaluator.evaluate_gnn_model(full_data)
    
    if gnn_results:
        print(f"\nGNN Results:")
        print(f"  Accuracy: {gnn_results['accuracy']:.4f}")
        print(f"  AUC: {gnn_results['auc']:.4f}")
        print(f"  F1-Score: {gnn_results['f1_score']:.4f}")
        print(f"  Inference Time: {gnn_results['avg_inference_time']:.4f} seconds")
        print(f"  Throughput: {gnn_results['throughput']:.1f} samples/second")
    
    return gnn_results

def test_debug_explorer():
    """Test the debug explorer with subgraph visualization"""
    print("\n" + "="*50)
    print("Testing Debug Explorer with Subgraph Visualization")
    print("="*50)
    
    # Create debug explorer
    explorer = GraphDebugExplorer(random_state=42)
    
    # Run exploration
    graph_data = explorer.run_complete_debug_exploration()
    
    return graph_data

def main():
    """Main test function"""
    print("Subgraph Inference Test")
    print("="*50)
    
    # Test evaluation
    gnn_results = test_subgraph_inference()
    
    # Test debug explorer
    graph_data = test_debug_explorer()
    
    print("\n" + "="*50)
    print("Test completed!")
    print("Check the ./outputs/ directory for visualization files.")
    
    return gnn_results, graph_data

if __name__ == '__main__':
    main() 