"""
Example usage of the Graph Debug Explorer
This script demonstrates how to explore and understand the graph structure used for GNN inference.
"""

from graph_debug_explorer import GraphDebugExplorer

def main():
    """Example usage of the graph debug explorer"""
    print("Graph Debug Explorer Example")
    print("="*50)
    
    # Create the explorer
    explorer = GraphDebugExplorer(random_state=42)
    
    # Load data and build graph
    print("Loading data and building graph...")
    graph_data = explorer.load_data_and_build_graph()
    
    # Explore different aspects of the graph
    print("\n1. Exploring graph structure...")
    explorer.explore_graph_structure()
    
    print("\n2. Exploring node features...")
    explorer.explore_node_features(num_samples=3)
    
    print("\n3. Exploring edge features...")
    explorer.explore_edge_features(num_samples=3)
    
    print("\n4. Showing inference samples...")
    explorer.show_inference_sample(num_samples=2)
    
    print("\n5. Creating visualizations...")
    explorer.visualize_feature_distributions()
    explorer.create_simple_graph_visualization(max_nodes=30)
    
    print("\nDebug exploration completed!")
    print("Check the ./outputs/ directory for visualization files.")
    
    return graph_data

if __name__ == '__main__':
    main() 