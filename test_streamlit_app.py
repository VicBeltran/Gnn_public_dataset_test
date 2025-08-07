#!/usr/bin/env python3
"""
Test script for the Streamlit GNN Inference Explorer app
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("✓ Streamlit imported successfully")
    except ImportError as e:
        print(f"✗ Streamlit import failed: {e}")
        return False
    
    try:
        import plotly.graph_objects as go
        print("✓ Plotly imported successfully")
    except ImportError as e:
        print(f"✗ Plotly import failed: {e}")
        return False
    
    try:
        from graph_debug_explorer import GraphDebugExplorer
        print("✓ GraphDebugExplorer imported successfully")
    except ImportError as e:
        print(f"✗ GraphDebugExplorer import failed: {e}")
        return False
    
    try:
        from gnn_wrapper import GNNTrainingWrapper
        print("✓ GNNTrainingWrapper imported successfully")
    except ImportError as e:
        print(f"✗ GNNTrainingWrapper import failed: {e}")
        return False
    
    try:
        from models import ImprovedEdgeGraphSAGE
        print("✓ Models imported successfully")
    except ImportError as e:
        print(f"✗ Models import failed: {e}")
        return False
    
    return True

def test_data_file():
    """Test if the required data file exists"""
    print("\nTesting data file...")
    
    if os.path.exists('./preprocessed_fraud_test.csv'):
        print("✓ preprocessed_fraud_test.csv found")
        return True
    else:
        print("✗ preprocessed_fraud_test.csv not found")
        print("  Please ensure the data file is in the current directory")
        return False

def test_graph_loading():
    """Test if graph data can be loaded"""
    print("\nTesting graph loading...")
    
    try:
        from graph_debug_explorer import GraphDebugExplorer
        
        # Create explorer instance
        explorer = GraphDebugExplorer(random_state=42)
        print("✓ GraphDebugExplorer instance created")
        
        # Try to load data (this might take a while)
        print("  Loading graph data (this may take a moment)...")
        graph_data = explorer.load_data_and_build_graph()
        
        if graph_data is not None:
            print(f"✓ Graph loaded successfully")
            print(f"  Nodes: {graph_data.num_nodes}")
            print(f"  Edges: {graph_data.num_edges}")
            print(f"  Node features: {graph_data.num_node_features}")
            print(f"  Edge features: {graph_data.edge_attr.shape[1]}")
            return True
        else:
            print("✗ Graph loading failed")
            return False
            
    except Exception as e:
        print(f"✗ Graph loading error: {e}")
        return False

def test_subgraph_sampling():
    """Test subgraph sampling functionality"""
    print("\nTesting subgraph sampling...")
    
    try:
        from graph_debug_explorer import GraphDebugExplorer
        
        # Load graph data
        explorer = GraphDebugExplorer(random_state=42)
        graph_data = explorer.load_data_and_build_graph()
        
        if graph_data is None:
            print("✗ Cannot test subgraph sampling - no graph data")
            return False
        
        # Test subgraph sampling
        target_edge_indices = [0, 100, 500]  # Test a few edges
        subgraph_data = explorer.gnn_wrapper.sample_subgraph_around_edges(
            graph_data, target_edge_indices, max_nodes=200
        )
        
        if subgraph_data is not None:
            print(f"✓ Subgraph sampling successful")
            print(f"  Subgraph nodes: {subgraph_data.num_nodes}")
            print(f"  Subgraph edges: {subgraph_data.num_edges}")
            return True
        else:
            print("✗ Subgraph sampling failed")
            return False
            
    except Exception as e:
        print(f"✗ Subgraph sampling error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("GNN Inference Explorer - App Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Data File Test", test_data_file),
        ("Graph Loading Test", test_graph_loading),
        ("Subgraph Sampling Test", test_subgraph_sampling)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The app should work correctly.")
        print("\nTo run the app:")
        print("  python run_streamlit_app.py")
        print("  or")
        print("  streamlit run streamlit_inference_app.py")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 