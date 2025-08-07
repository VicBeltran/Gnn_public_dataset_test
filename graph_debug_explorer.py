"""
Graph Debug Explorer for GNN Inference
A focused tool for exploring and debugging the graph structure used in GNN inference.
Shows detailed node and edge properties, feature distributions, and sample inference data.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
from datetime import datetime

from models import ImprovedEdgeGraphSAGE, HeterogeneousEdgeGraphSAGE
from gnn_wrapper import GNNTrainingWrapper

warnings.filterwarnings('ignore')

class GraphDebugExplorer:
    """Focused tool for debugging GNN graph structure and inference"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.gnn_wrapper = None
        self.graph_data = None
        self.df = None
        self.source_node_features = None
        self.target_node_features = None
        self.edge_features = None
        
        # Set random seed
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
    def load_data_and_build_graph(self):
        """Load data and build graph for debugging"""
        print("Loading data and building graph for debugging...")
        
        # Load the preprocessed data
        self.df = pd.read_csv('./preprocessed_fraud_test.csv')
        
        # Define features (same as in evaluate_models.py)
        self.source_node_features = ['gender', 'street', 'city',
                                   'state', 'zip', 'lat', 'long', 'city_pop', 'job']
        
        self.target_node_features = ['merch_lat', 'merch_long']
        
        self.edge_features = ['amt', 'category', 'trans_year', 'trans_month', 
                             'trans_day', 'trans_hour', 'trans_dow']
        
        print(f"Data loaded: {len(self.df)} transactions")
        print(f"Source node features: {len(self.source_node_features)} features")
        print(f"Target node features: {len(self.target_node_features)} features")
        print(f"Edge features: {len(self.edge_features)} features")
        
        # Create GNN wrapper and build graph
        self.gnn_wrapper = GNNTrainingWrapper(random_state=self.random_state)
        
        self.graph_data = self.gnn_wrapper.build_graph_from_tabular(
            df=self.df,
            node_features=self.source_node_features + self.target_node_features + self.edge_features,
            edge_features=self.edge_features,
            target_col='is_fraud',
            source_node_col='cc_num',
            target_node_col='merchant',
            source_node_features=self.source_node_features,
            target_node_features=self.target_node_features
        )
        
        print(f"Graph built: {self.graph_data.num_nodes} nodes, {self.graph_data.num_edges} edges")
        print(f"Fraud rate: {self.graph_data.edge_labels.float().mean():.4f}")
        
        return self.graph_data
    
    def explore_graph_structure(self):
        """Explore the basic graph structure"""
        print("\n" + "="*60)
        print("GRAPH STRUCTURE EXPLORATION")
        print("="*60)
        
        if self.graph_data is None:
            print("No graph data loaded")
            return
        
        # Basic graph statistics
        num_nodes = self.graph_data.num_nodes
        num_edges = self.graph_data.num_edges
        num_source_nodes = len(self.df['cc_num'].unique())
        num_target_nodes = len(self.df['merchant'].unique())
        
        print(f"Graph Statistics:")
        print(f"  Total Nodes: {num_nodes}")
        print(f"  Source Nodes (Customers): {num_source_nodes}")
        print(f"  Target Nodes (Merchants): {num_target_nodes}")
        print(f"  Total Edges (Transactions): {num_edges}")
        print(f"  Fraud Rate: {self.graph_data.edge_labels.float().mean():.4f}")
        
        # Node feature dimensions
        print(f"\nFeature Dimensions:")
        print(f"  Node Features: {self.graph_data.num_node_features}")
        print(f"  Edge Features: {self.graph_data.edge_attr.shape[1]}")
        
        # Edge direction analysis
        edge_index = self.graph_data.edge_index.cpu().numpy()
        source_nodes = edge_index[0, :]
        target_nodes = edge_index[1, :]
        
        print(f"\nEdge Direction Analysis:")
        print(f"  Source nodes range: {source_nodes.min()} to {source_nodes.max()}")
        print(f"  Target nodes range: {target_nodes.min()} to {target_nodes.max()}")
        print(f"  Unique source nodes: {len(np.unique(source_nodes))}")
        print(f"  Unique target nodes: {len(np.unique(target_nodes))}")
        
        # Degree analysis
        from collections import Counter
        source_degrees = Counter(source_nodes)
        target_degrees = Counter(target_nodes)
        
        print(f"\nDegree Analysis:")
        print(f"  Source node degrees - Min: {min(source_degrees.values())}, Max: {max(source_degrees.values())}")
        print(f"  Target node degrees - Min: {min(target_degrees.values())}, Max: {max(target_degrees.values())}")
        
        # Fraud distribution
        edge_labels = self.graph_data.edge_labels.cpu().numpy()
        fraud_edges = edge_labels == 1
        normal_edges = edge_labels == 0
        
        print(f"\nFraud Distribution:")
        print(f"  Fraud edges: {fraud_edges.sum()} ({fraud_edges.sum()/len(edge_labels):.3f})")
        print(f"  Normal edges: {normal_edges.sum()} ({normal_edges.sum()/len(edge_labels):.3f})")
    
    def explore_node_features(self, num_samples=5):
        """Explore node features in detail"""
        print("\n" + "="*60)
        print("NODE FEATURES EXPLORATION")
        print("="*60)
        
        if self.graph_data is None:
            print("No graph data loaded")
            return
        
        node_features = self.graph_data.x.cpu().numpy()
        num_source_nodes = len(self.df['cc_num'].unique())
        
        print(f"Node Features Shape: {node_features.shape}")
        print(f"Source nodes (0 to {num_source_nodes-1}): Customer features")
        print(f"Target nodes ({num_source_nodes} to {node_features.shape[0]-1}): Merchant features")
        
        # Sample source nodes (customers)
        source_samples = np.random.choice(num_source_nodes, min(num_samples, num_source_nodes), replace=False)
        
        print(f"\nSample Source Nodes (Customers):")
        for i, node_id in enumerate(source_samples):
            features = node_features[node_id]
            print(f"\nCustomer Node {node_id}:")
            print(f"  Feature vector: {features}")
            print(f"  Feature statistics:")
            print(f"    Mean: {np.mean(features):.4f}")
            print(f"    Std: {np.std(features):.4f}")
            print(f"    Min: {np.min(features):.4f}")
            print(f"    Max: {np.max(features):.4f}")
            
            # Show feature names and values
            print(f"  Feature breakdown:")
            for j, feature_name in enumerate(self.source_node_features):
                if j < len(features):
                    print(f"    {feature_name}: {features[j]:.4f}")
        
        # Sample target nodes (merchants)
        target_samples = np.random.choice(
            range(num_source_nodes, node_features.shape[0]), 
            min(num_samples, node_features.shape[0] - num_source_nodes), 
            replace=False
        )
        
        print(f"\nSample Target Nodes (Merchants):")
        for i, node_id in enumerate(target_samples):
            features = node_features[node_id]
            print(f"\nMerchant Node {node_id}:")
            print(f"  Feature vector: {features}")
            print(f"  Feature statistics:")
            print(f"    Mean: {np.mean(features):.4f}")
            print(f"    Std: {np.std(features):.4f}")
            print(f"    Min: {np.min(features):.4f}")
            print(f"    Max: {np.max(features):.4f}")
            
            # Show feature names and values
            print(f"  Feature breakdown:")
            for j, feature_name in enumerate(self.target_node_features):
                if j < len(features):
                    print(f"    {feature_name}: {features[j]:.4f}")
    
    def explore_edge_features(self, num_samples=5):
        """Explore edge features in detail"""
        print("\n" + "="*60)
        print("EDGE FEATURES EXPLORATION")
        print("="*60)
        
        if self.graph_data is None:
            print("No graph data loaded")
            return
        
        edge_features = self.graph_data.edge_attr.cpu().numpy()
        edge_labels = self.graph_data.edge_labels.cpu().numpy()
        edge_index = self.graph_data.edge_index.cpu().numpy()
        
        print(f"Edge Features Shape: {edge_features.shape}")
        print(f"Edge Labels Shape: {edge_labels.shape}")
        
        # Sample edges
        sample_indices = np.random.choice(len(edge_features), min(num_samples, len(edge_features)), replace=False)
        
        print(f"\nSample Edges (Transactions):")
        for i, edge_id in enumerate(sample_indices):
            features = edge_features[edge_id]
            is_fraud = edge_labels[edge_id]
            source = edge_index[0, edge_id]
            target = edge_index[1, edge_id]
            
            print(f"\nEdge {edge_id} (Source {source} → Target {target}):")
            print(f"  Is Fraud: {is_fraud}")
            print(f"  Feature vector: {features}")
            print(f"  Feature statistics:")
            print(f"    Mean: {np.mean(features):.4f}")
            print(f"    Std: {np.std(features):.4f}")
            print(f"    Min: {np.min(features):.4f}")
            print(f"    Max: {np.max(features):.4f}")
            
            # Show feature names and values
            print(f"  Feature breakdown:")
            for j, feature_name in enumerate(self.edge_features):
                if j < len(features):
                    print(f"    {feature_name}: {features[j]:.4f}")
        
        # Analyze fraud vs normal edge features
        fraud_edges = edge_features[edge_labels == 1]
        normal_edges = edge_features[edge_labels == 0]
        
        print(f"\nFeature Analysis by Fraud Status:")
        print(f"  Fraud edges: {fraud_edges.shape[0]} samples")
        print(f"  Normal edges: {normal_edges.shape[0]} samples")
        
        if len(fraud_edges) > 0 and len(normal_edges) > 0:
            print(f"  Fraud edge feature means: {np.mean(fraud_edges, axis=0)}")
            print(f"  Normal edge feature means: {np.mean(normal_edges, axis=0)}")
            
            # Show feature-wise differences
            print(f"  Feature-wise differences (Fraud - Normal):")
            for j, feature_name in enumerate(self.edge_features):
                if j < fraud_edges.shape[1]:
                    fraud_mean = np.mean(fraud_edges[:, j])
                    normal_mean = np.mean(normal_edges[:, j])
                    diff = fraud_mean - normal_mean
                    print(f"    {feature_name}: {diff:.4f} ({fraud_mean:.4f} vs {normal_mean:.4f})")
    
    def show_inference_sample(self, num_samples=3):
        """Show detailed samples of what the GNN sees during inference"""
        print("\n" + "="*60)
        print("INFERENCE SAMPLE EXPLORATION")
        print("="*60)
        
        if self.graph_data is None:
            print("No graph data loaded")
            return
        
        # Sample edges for inference
        sample_indices = np.random.choice(
            self.graph_data.num_edges, 
            min(num_samples, self.graph_data.num_edges), 
            replace=False
        )
        
        edge_index = self.graph_data.edge_index.cpu().numpy()
        edge_attr = self.graph_data.edge_attr.cpu().numpy()
        edge_labels = self.graph_data.edge_labels.cpu().numpy()
        node_features = self.graph_data.x.cpu().numpy()
        
        print(f"Showing {len(sample_indices)} inference samples:")
        
        for i, edge_id in enumerate(sample_indices):
            source = edge_index[0, edge_id]
            target = edge_index[1, edge_id]
            edge_features = edge_attr[edge_id]
            is_fraud = edge_labels[edge_id]
            
            source_features = node_features[source]
            target_features = node_features[target]
            
            print(f"\n{'='*40}")
            print(f"INFERENCE SAMPLE {i+1} (Edge {edge_id})")
            print(f"{'='*40}")
            
            print(f"Edge Information:")
            print(f"  Source Node: {source}")
            print(f"  Target Node: {target}")
            print(f"  Is Fraud: {is_fraud}")
            print(f"  Edge Features: {edge_features}")
            
            print(f"\nSource Node Features (Customer):")
            print(f"  Feature vector: {source_features}")
            print(f"  Feature breakdown:")
            for j, feature_name in enumerate(self.source_node_features):
                if j < len(source_features):
                    print(f"    {feature_name}: {source_features[j]:.4f}")
            
            print(f"\nTarget Node Features (Merchant):")
            print(f"  Feature vector: {target_features}")
            print(f"  Feature breakdown:")
            for j, feature_name in enumerate(self.target_node_features):
                if j < len(target_features):
                    print(f"    {feature_name}: {target_features[j]:.4f}")
            
            # Show what the GNN model would receive
            print(f"\nGNN Model Input:")
            print(f"  Node features shape: {node_features.shape}")
            print(f"  Edge features shape: {edge_attr.shape}")
            print(f"  Edge index shape: {edge_index.shape}")
            print(f"  For this specific edge:")
            print(f"    Source node features: {source_features}")
            print(f"    Target node features: {target_features}")
            print(f"    Edge features: {edge_features}")
            print(f"    Target label: {is_fraud}")
    
    def visualize_feature_distributions(self):
        """Create visualizations of feature distributions"""
        print("\n" + "="*60)
        print("FEATURE DISTRIBUTION VISUALIZATION")
        print("="*60)
        
        if self.graph_data is None:
            print("No graph data loaded")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('GNN Graph Feature Distributions', fontsize=16, fontweight='bold')
        
        node_features = self.graph_data.x.cpu().numpy()
        edge_features = self.graph_data.edge_attr.cpu().numpy()
        edge_labels = self.graph_data.edge_labels.cpu().numpy()
        
        # 1. Node feature distribution
        ax1 = axes[0, 0]
        ax1.hist(node_features.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Node Feature Distribution')
        ax1.set_xlabel('Feature Values')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # 2. Edge feature distribution
        ax2 = axes[0, 1]
        ax2.hist(edge_features.flatten(), bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_title('Edge Feature Distribution')
        ax2.set_xlabel('Feature Values')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # 3. Fraud vs Normal edge features
        ax3 = axes[1, 0]
        fraud_edges = edge_features[edge_labels == 1]
        normal_edges = edge_features[edge_labels == 0]
        
        if len(fraud_edges) > 0:
            ax3.hist(fraud_edges.flatten(), bins=30, alpha=0.7, label='Fraud', color='red')
        if len(normal_edges) > 0:
            ax3.hist(normal_edges.flatten(), bins=30, alpha=0.7, label='Normal', color='blue')
        
        ax3.set_title('Edge Features: Fraud vs Normal')
        ax3.set_xlabel('Feature Values')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Feature correlation heatmap (sample of features)
        ax4 = axes[1, 1]
        sample_size = min(1000, len(edge_features))
        sample_indices = np.random.choice(len(edge_features), sample_size, replace=False)
        sample_features = edge_features[sample_indices]
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(sample_features.T)
        
        # Create heatmap
        im = ax4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax4.set_title('Edge Feature Correlation Matrix')
        ax4.set_xlabel('Feature Index')
        ax4.set_ylabel('Feature Index')
        
        # Add colorbar
        plt.colorbar(im, ax=ax4)
        
        # Add feature names as tick labels
        if len(self.edge_features) <= 10:  # Only if we have few features
            ax4.set_xticks(range(len(self.edge_features)))
            ax4.set_yticks(range(len(self.edge_features)))
            ax4.set_xticklabels(self.edge_features, rotation=45)
            ax4.set_yticklabels(self.edge_features)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f'./outputs/feature_distributions_{timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"✓ Feature distribution plot saved to {plot_filename}")
        
        plt.show()
    
    def create_simple_graph_visualization(self, max_nodes=50):
        """Create a simple graph visualization showing the subgraph used for inference"""
        print("\n" + "="*60)
        print("INFERENCE SUBGRAPH VISUALIZATION")
        print("="*60)
        
        if self.graph_data is None:
            print("No graph data loaded")
            return
        
        # Configure subgraph sampling parameters to match inference settings
        self.gnn_wrapper.max_subgraph_size = 500  # Max nodes in subgraph
        self.gnn_wrapper.num_hops = 2  # 2-hop neighborhood for realistic inference
        
        print(f"Using subgraph sampling with {self.gnn_wrapper.num_hops} hops and max {self.gnn_wrapper.max_subgraph_size} nodes")
        
        # Sample a few target edges for visualization
        num_target_edges = min(5, self.graph_data.num_edges)
        target_edge_indices = np.random.choice(
            self.graph_data.num_edges, 
            num_target_edges, 
            replace=False
        )
        
        print(f"Sampling subgraph around {num_target_edges} target edges...")
        
        # Sample subgraph around these edges (this is what would be used in inference)
        inference_subgraph = self.gnn_wrapper.sample_subgraph_around_edges(
            self.graph_data, target_edge_indices, max_nodes=self.gnn_wrapper.max_subgraph_size
        )
        
        if inference_subgraph is None:
            print("Subgraph sampling failed")
            return
        
        print(f"Inference subgraph: {inference_subgraph.num_nodes} nodes, {inference_subgraph.num_edges} edges")
        
        # Convert to NetworkX for visualization
        inference_nx = nx.DiGraph()
        
        # Add nodes with their original IDs for better understanding
        num_source_nodes = len(self.df['cc_num'].unique())
        for i in range(inference_subgraph.num_nodes):
            # Determine if this is a source or target node based on original mapping
            if hasattr(inference_subgraph, 'node_mask') and inference_subgraph.node_mask is not None:
                # Get the original node indices that are included in the subgraph
                original_node_indices = torch.where(inference_subgraph.node_mask)[0]
                if i < len(original_node_indices):
                    original_node_id = original_node_indices[i].item()
                    node_type = 'source' if original_node_id < num_source_nodes else 'target'
                    # Store original ID for reference
                    inference_nx.add_node(i, type=node_type, original_id=original_node_id)
                else:
                    # Fallback if index is out of bounds
                    node_type = 'source' if i < num_source_nodes else 'target'
                    inference_nx.add_node(i, type=node_type, original_id=i)
            else:
                node_type = 'source' if i < num_source_nodes else 'target'
                inference_nx.add_node(i, type=node_type, original_id=i)
        
        # Add edges
        edge_index = inference_subgraph.edge_index.cpu().numpy()
        edge_labels = inference_subgraph.edge_labels.cpu().numpy()
        
        for i in range(inference_subgraph.num_edges):
            source = edge_index[0, i]
            target = edge_index[1, i]
            is_fraud = edge_labels[i]
            inference_nx.add_edge(source, target, is_fraud=is_fraud)
        
        # Create visualization
        plt.figure(figsize=(14, 10))
        
        # Create layout
        pos = nx.spring_layout(inference_nx, k=1, iterations=50, seed=self.random_state)
        
        # Separate nodes and edges
        source_nodes = [n for n, d in inference_nx.nodes(data=True) if d['type'] == 'source']
        target_nodes = [n for n, d in inference_nx.nodes(data=True) if d['type'] == 'target']
        
        fraud_edges = [(u, v) for u, v, d in inference_nx.edges(data=True) if d['is_fraud'] == 1]
        normal_edges = [(u, v) for u, v, d in inference_nx.edges(data=True) if d['is_fraud'] == 0]
        
        # Draw graph
        nx.draw_networkx_nodes(inference_nx, pos, nodelist=source_nodes, 
                              node_color='red', node_size=400, alpha=0.8, label='Customers')
        nx.draw_networkx_nodes(inference_nx, pos, nodelist=target_nodes, 
                              node_color='blue', node_size=400, alpha=0.8, label='Merchants')
        
        nx.draw_networkx_edges(inference_nx, pos, edgelist=normal_edges, 
                              edge_color='green', width=1, alpha=0.6, label='Normal Transactions')
        nx.draw_networkx_edges(inference_nx, pos, edgelist=fraud_edges, 
                              edge_color='red', width=3, alpha=0.8, label='Fraudulent Transactions')
        
        # Add labels for nodes (show original IDs)
        labels = {}
        for n, d in inference_nx.nodes(data=True):
            if 'original_id' in d:
                labels[n] = f"{d['original_id']}"
        
        nx.draw_networkx_labels(inference_nx, pos, labels, font_size=8)
        
        plt.title(f'Inference Subgraph ({inference_nx.number_of_nodes()} nodes, {inference_nx.number_of_edges()} edges)\n'
                 f'2-hop neighborhood around {num_target_edges} target transactions')
        plt.legend()
        plt.axis('off')
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f'./outputs/inference_subgraph_{timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"✓ Inference subgraph visualization saved to {plot_filename}")
        
        plt.show()
        
        # Print subgraph statistics
        print(f"\nInference Subgraph Statistics:")
        print(f"  Total nodes: {inference_nx.number_of_nodes()}")
        print(f"  Customer nodes: {len(source_nodes)}")
        print(f"  Merchant nodes: {len(target_nodes)}")
        print(f"  Total edges: {inference_nx.number_of_edges()}")
        print(f"  Fraudulent edges: {len(fraud_edges)}")
        print(f"  Normal edges: {len(normal_edges)}")
        print(f"  Fraud rate in subgraph: {len(fraud_edges)/inference_nx.number_of_edges():.3f}")
        
        return inference_nx
    
    def demonstrate_inference_process(self, num_target_edges=3):
        """Demonstrate the actual inference process with subgraph sampling"""
        print("\n" + "="*60)
        print("INFERENCE PROCESS DEMONSTRATION")
        print("="*60)
        
        if self.graph_data is None:
            print("No graph data loaded")
            return
        
        # Configure subgraph sampling parameters
        self.gnn_wrapper.max_subgraph_size = 500
        self.gnn_wrapper.num_hops = 2
        
        # Sample target edges for inference demonstration
        target_edge_indices = np.random.choice(
            self.graph_data.num_edges, 
            min(num_target_edges, self.graph_data.num_edges), 
            replace=False
        )
        
        print(f"Demonstrating inference for {len(target_edge_indices)} target edges...")
        
        # Show the full graph statistics
        print(f"\nFull Graph Statistics:")
        print(f"  Total nodes: {self.graph_data.num_nodes}")
        print(f"  Total edges: {self.graph_data.num_edges}")
        print(f"  Fraud rate: {self.graph_data.edge_labels.float().mean():.3f}")
        
        # Sample subgraph for inference
        print(f"\nSampling subgraph for inference...")
        inference_subgraph = self.gnn_wrapper.sample_subgraph_around_edges(
            self.graph_data, target_edge_indices, max_nodes=self.gnn_wrapper.max_subgraph_size
        )
        
        if inference_subgraph is None:
            print("Subgraph sampling failed")
            return
        
        print(f"Inference Subgraph Statistics:")
        print(f"  Nodes: {inference_subgraph.num_nodes} (reduced from {self.graph_data.num_nodes})")
        print(f"  Edges: {inference_subgraph.num_edges} (reduced from {self.graph_data.num_edges})")
        print(f"  Reduction factor: {self.graph_data.num_nodes/inference_subgraph.num_nodes:.1f}x nodes, {self.graph_data.num_edges/inference_subgraph.num_edges:.1f}x edges")
        
        # Show target edges in the subgraph
        edge_index = self.graph_data.edge_index.cpu().numpy()
        edge_labels = self.graph_data.edge_labels.cpu().numpy()
        
        print(f"\nTarget Edges for Inference:")
        for i, edge_idx in enumerate(target_edge_indices):
            source = edge_index[0, edge_idx]
            target = edge_index[1, edge_idx]
            is_fraud = edge_labels[edge_idx]
            
            print(f"  Edge {i+1}: Customer {source} → Merchant {target} (Fraud: {is_fraud})")
        
        # Show what the model would receive
        print(f"\nModel Input for Inference:")
        print(f"  Node features shape: {inference_subgraph.x.shape}")
        print(f"  Edge features shape: {inference_subgraph.edge_attr.shape}")
        print(f"  Edge index shape: {inference_subgraph.edge_index.shape}")
        
        # Calculate memory usage
        node_memory = inference_subgraph.x.numel() * 4  # 4 bytes per float32
        edge_memory = inference_subgraph.edge_attr.numel() * 4
        index_memory = inference_subgraph.edge_index.numel() * 8  # 8 bytes per int64
        
        total_memory_mb = (node_memory + edge_memory + index_memory) / (1024 * 1024)
        
        print(f"\nMemory Usage:")
        print(f"  Node features: {node_memory / 1024:.1f} KB")
        print(f"  Edge features: {edge_memory / 1024:.1f} KB")
        print(f"  Edge indices: {index_memory / 1024:.1f} KB")
        print(f"  Total: {total_memory_mb:.2f} MB")
        
        # Show inference time estimation
        print(f"\nInference Time Estimation:")
        print(f"  Subgraph size: {inference_subgraph.num_nodes} nodes, {inference_subgraph.num_edges} edges")
        print(f"  Typical GNN inference: ~1-10ms per subgraph")
        print(f"  Real-time capable: Yes (subgraph sampling enables real-time inference)")
        
        return inference_subgraph
    
    def run_complete_debug_exploration(self):
        """Run the complete debug exploration"""
        print("Starting Complete Graph Debug Exploration")
        print("="*60)
        
        # Load data and build graph
        self.load_data_and_build_graph()
        
        # Run all exploration methods
        self.explore_graph_structure()
        self.explore_node_features()
        self.explore_edge_features()
        self.show_inference_sample()
        self.visualize_feature_distributions()
        self.create_simple_graph_visualization()
        self.demonstrate_inference_process()
        
        print("\n" + "="*60)
        print("DEBUG EXPLORATION COMPLETED")
        print("="*60)
        print("Check the ./outputs/ directory for visualization files.")
        
        return self.graph_data

def main():
    """Main function to run the debug exploration"""
    explorer = GraphDebugExplorer(random_state=42)
    graph_data = explorer.run_complete_debug_exploration()
    
    return graph_data

if __name__ == '__main__':
    main() 