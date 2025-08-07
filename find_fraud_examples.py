#!/usr/bin/env python3
"""
Script to find fraud examples in the dataset
"""

import os
import pandas as pd
import numpy as np
from graph_debug_explorer import GraphDebugExplorer

def find_fraud_examples():
    """Find examples of fraudulent transactions in the dataset"""
    print("🔍 Finding fraud examples in the dataset...")
    
    # Check if data file exists
    if not os.path.exists('./preprocessed_fraud_test.csv'):
        print("Error: preprocessed_fraud_test.csv not found")
        return
    
    # Load the data
    df = pd.read_csv('./preprocessed_fraud_test.csv')
    print(f"Dataset loaded: {len(df)} transactions")
    
    # Check if 'is_fraud' column exists
    if 'is_fraud' not in df.columns:
        print("Error: 'is_fraud' column not found in dataset")
        print("Available columns:", list(df.columns))
        return
    
    # Find fraud examples
    fraud_df = df[df['is_fraud'] == 1]
    normal_df = df[df['is_fraud'] == 0]
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total transactions: {len(df)}")
    print(f"   Fraudulent transactions: {len(fraud_df)} ({len(fraud_df)/len(df)*100:.2f}%)")
    print(f"   Normal transactions: {len(normal_df)} ({len(normal_df)/len(df)*100:.2f}%)")
    
    if len(fraud_df) == 0:
        print("\n❌ No fraudulent transactions found in the dataset!")
        return
    
    # Show some fraud examples
    print(f"\n🎯 Fraud Examples (first 10):")
    print("=" * 80)
    
    fraud_examples = fraud_df.head(10)
    for idx, row in fraud_examples.iterrows():
        print(f"Edge ID: {idx}")
        print(f"  Customer ID: {row.get('customer_id', 'N/A')}")
        print(f"  Merchant ID: {row.get('merchant_id', 'N/A')}")
        print(f"  Amount: {row.get('amt', 'N/A')}")
        print(f"  Category: {row.get('category', 'N/A')}")
        print(f"  Transaction Date: {row.get('trans_date', 'N/A')}")
        print("-" * 40)
    
    # Get edge indices for fraud examples
    fraud_indices = fraud_df.index.tolist()
    
    print(f"\n💡 Recommended Edge IDs for Streamlit Inference:")
    print("=" * 50)
    
    # Show first 5 fraud examples with their edge indices
    for i, edge_id in enumerate(fraud_indices[:5]):
        row = fraud_df.loc[edge_id]
        print(f"Edge ID {edge_id}:")
        print(f"  - Customer: {row.get('customer_id', 'N/A')}")
        print(f"  - Merchant: {row.get('merchant_id', 'N/A')}")
        print(f"  - Amount: ${row.get('amt', 0):.2f}")
        print(f"  - Category: {row.get('category', 'N/A')}")
        print()
    
    # Also show some normal examples for comparison
    print(f"\n📋 Normal Transaction Examples (for comparison):")
    print("=" * 50)
    
    normal_examples = normal_df.head(3)
    for idx, row in normal_examples.iterrows():
        print(f"Edge ID {idx}:")
        print(f"  - Customer: {row.get('customer_id', 'N/A')}")
        print(f"  - Merchant: {row.get('merchant_id', 'N/A')}")
        print(f"  - Amount: ${row.get('amt', 0):.2f}")
        print(f"  - Category: {row.get('category', 'N/A')}")
        print()
    
    # Test with graph data
    print(f"\n🔬 Testing with Graph Data:")
    print("=" * 30)
    
    try:
        explorer = GraphDebugExplorer(random_state=42)
        graph_data = explorer.load_data_and_build_graph()
        
        if graph_data is not None:
            edge_labels = graph_data.edge_labels.cpu().numpy()
            fraud_edge_indices = np.where(edge_labels == 1)[0]
            normal_edge_indices = np.where(edge_labels == 0)[0]
            
            print(f"Graph loaded successfully!")
            print(f"  Total edges in graph: {len(edge_labels)}")
            print(f"  Fraudulent edges: {len(fraud_edge_indices)}")
            print(f"  Normal edges: {len(normal_edge_indices)}")
            
            if len(fraud_edge_indices) > 0:
                print(f"\n🎯 Fraud Edge Indices for Streamlit (first 10):")
                for i, edge_idx in enumerate(fraud_edge_indices[:10]):
                    print(f"  Edge {edge_idx}")
                
                print(f"\n✅ RECOMMENDED: Use Edge {fraud_edge_indices[0]} for fraud inference testing")
            else:
                print("❌ No fraudulent edges found in graph data")
                
        else:
            print("❌ Failed to load graph data")
            
    except Exception as e:
        print(f"❌ Error loading graph data: {e}")
    
    print(f"\n📝 Instructions:")
    print("1. Use one of the fraud edge IDs above in the Streamlit app")
    print("2. The edge should show as fraudulent in the visualization")
    print("3. The inference should predict a high fraud probability")

if __name__ == "__main__":
    find_fraud_examples() 