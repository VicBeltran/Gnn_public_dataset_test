"""
Interactive Graph Explorer for GNN Inference Debugging
This tool helps visualize and explore the graph structure used for GNN inference,
showing node properties, edge properties, and allowing sample highlighting.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, Slider, CheckButtons
import seaborn as sns
from collections import defaultdict
import warnings
from datetime import datetime

from models import ImprovedEdgeGraphSAGE, HeterogeneousEdgeGraphSAGE
from gnn_wrapper import GNNTrainingWrapper

warnings.filterwarnings('ignore')

class GraphExplorer:
    """Interactive tool to explore GNN graph structure and properties"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.gnn_wrapper = None
        self.graph_data = None
        self.networkx_graph = None
        self.df = None
        self.node_features = None
        self.edge_features = None
        self.source_node_features = None
        self.target_node_features = None
        
        # Visualization settings
        self.node_size = 300
        self.edge_width = 1.0
        self.font_size = 8
        self.fig_size = (16, 12)
        
        # Color schemes
        self.colors = {
            'source_node': '#FF6B35',
            'target_node': '#4ECDC4', 
            'fraud_edge': '#E74C3C',
            'normal_edge': '#2ECC71',
            'highlight': '#F39C12',
            'background': '#ECF0F1'
        }
        
        # Interactive elements
        self.fig = None
        self.ax = None
        self.highlighted_nodes = set()
        self.highlighted_edges = set()
        self.selected_edge = None
        
    def load_data_and_build_graph(self):
        """Load data and build graph for exploration"""
        print("Loading data and building graph...")
        
        # Load the preprocessed data
        self.df = pd.read_csv('./preprocessed_fraud_test.csv')
        
        # Define features (same as in evaluate_models.py)
        self.source_node_features = ['gender', 'street', 'city',
                                   'state', 'zip', 'lat', 'long', 'city_pop', 'job']
        
        self.target_node_features = ['merch_lat', 'merch_long']
        
        self.edge_features = ['amt', 'category', 'trans_year', 'trans_month', 
                             'trans_day', 'trans_hour', 'trans_dow']
        
        # Combined features for compatibility
        self.node_features = self.source_node_features + self.target_node_features + self.edge_features
        
        print(f"Data loaded: {len(self.df)} transactions")
        print(f"Source node features: {len(self.source_node_features)} features")
        print(f"Target node features: {len(self.target_node_features)} features")
        print(f"Edge features: {len(self.edge_features)} features")
        
        # Create GNN wrapper and build graph
        self.gnn_wrapper = GNNTrainingWrapper(random_state=self.random_state)
        
        self.graph_data = self.gnn_wrapper.build_graph_from_tabular(
            df=self.df,
            node_features=self.node_features,
            edge_features=self.edge_features,
            target_col='is_fraud',
            source_node_col='cc_num',
            target_node_col='merchant',
            source_node_features=self.source_node_features,
            target_node_features=self.target_node_features
        )
        
        # Convert to NetworkX for easier visualization
        self._convert_to_networkx()
        
        print(f"Graph built: {self.graph_data.num_nodes} nodes, {self.graph_data.num_edges} edges")
        print(f"Fraud rate: {self.graph_data.edge_labels.float().mean():.4f}")
        
        return self.graph_data
    
    def _convert_to_networkx(self):
        """Convert PyG graph to NetworkX for visualization"""
        print("Converting to NetworkX graph...")
        
        # Create NetworkX graph
        self.networkx_graph = nx.DiGraph()
        
        # Add nodes with properties
        num_source_nodes = len(self.df['cc_num'].unique())
        
        for i in range(self.graph_data.num_nodes):
            node_type = 'source' if i < num_source_nodes else 'target'
            node_id = i
            
            # Get node features
            node_features = self.graph_data.x[i].cpu().numpy()
            
            # Add node to NetworkX graph
            self.networkx_graph.add_node(node_id, 
                                       type=node_type,
                                       features=node_features,
                                       original_id=i)
        
        # Add edges with properties
        edge_index = self.graph_data.edge_index.cpu().numpy()
        edge_attr = self.graph_data.edge_attr.cpu().numpy()
        edge_labels = self.graph_data.edge_labels.cpu().numpy()
        
        for i in range(self.graph_data.num_edges):
            source = edge_index[0, i]
            target = edge_index[1, i]
            edge_features = edge_attr[i]
            is_fraud = edge_labels[i]
            
            # Add edge to NetworkX graph
            self.networkx_graph.add_edge(source, target,
                                       features=edge_features,
                                       is_fraud=is_fraud,
                                       edge_id=i)
        
        print(f"NetworkX graph created: {self.networkx_graph.number_of_nodes()} nodes, {self.networkx_graph.number_of_edges()} edges")
    
    def get_sample_subgraph(self, num_nodes=50, num_edges=100):
        """Get a sample subgraph for visualization"""
        print(f"Sampling subgraph with max {num_nodes} nodes and {num_edges} edges...")
        
        # Sample random edges
        edge_indices = np.random.choice(
            self.graph_data.num_edges, 
            size=min(num_edges, self.graph_data.num_edges), 
            replace=False
        )
        
        # Sample subgraph around these edges
        sample_data = self.gnn_wrapper.sample_subgraph_around_edges(
            self.graph_data, edge_indices, max_nodes=num_nodes
        )
        
        if sample_data is None:
            print("Subgraph sampling failed, using full graph")
            return self.graph_data
        
        print(f"Sample subgraph: {sample_data.num_nodes} nodes, {sample_data.num_edges} edges")
        return sample_data
    
    def create_interactive_visualization(self, sample_data=None):
        """Create interactive visualization of the graph"""
        if sample_data is None:
            sample_data = self.get_sample_subgraph()
        
        # Convert sample to NetworkX
        sample_nx = self._convert_sample_to_networkx(sample_data)
        
        # Create the main figure
        self.fig, self.ax = plt.subplots(figsize=self.fig_size)
        self.fig.suptitle('Interactive GNN Graph Explorer', fontsize=16, fontweight='bold')
        
        # Create the graph layout
        pos = nx.spring_layout(sample_nx, k=1, iterations=50, seed=self.random_state)
        
        # Draw the graph
        self._draw_graph(sample_nx, pos)
        
        # Add interactive controls
        self._add_controls(sample_nx, pos)
        
        # Add information panel
        self._add_info_panel(sample_data)
        
        plt.tight_layout()
        plt.show()
        
        return sample_nx, pos
    
    def _convert_sample_to_networkx(self, sample_data):
        """Convert sample PyG data to NetworkX graph"""
        sample_nx = nx.DiGraph()
        
        # Add nodes
        num_source_nodes = len(self.df['cc_num'].unique())
        
        for i in range(sample_data.num_nodes):
            node_type = 'source' if i < num_source_nodes else 'target'
            node_features = sample_data.x[i].cpu().numpy()
            
            sample_nx.add_node(i, 
                             type=node_type,
                             features=node_features,
                             original_id=i)
        
        # Add edges
        edge_index = sample_data.edge_index.cpu().numpy()
        edge_attr = sample_data.edge_attr.cpu().numpy()
        edge_labels = sample_data.edge_labels.cpu().numpy()
        
        for i in range(sample_data.num_edges):
            source = edge_index[0, i]
            target = edge_index[1, i]
            edge_features = edge_attr[i]
            is_fraud = edge_labels[i]
            
            sample_nx.add_edge(source, target,
                             features=edge_features,
                             is_fraud=is_fraud,
                             edge_id=i)
        
        return sample_nx
    
    def _draw_graph(self, graph, pos):
        """Draw the graph with proper styling"""
        # Clear the axis
        self.ax.clear()
        
        # Separate nodes by type
        source_nodes = [n for n, d in graph.nodes(data=True) if d['type'] == 'source']
        target_nodes = [n for n, d in graph.nodes(data=True) if d['type'] == 'target']
        
        # Separate edges by fraud status
        fraud_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['is_fraud'] == 1]
        normal_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['is_fraud'] == 0]
        
        # Draw edges
        nx.draw_networkx_edges(graph, pos, edgelist=normal_edges, 
                              edge_color=self.colors['normal_edge'], 
                              width=self.edge_width, alpha=0.6, ax=self.ax)
        nx.draw_networkx_edges(graph, pos, edgelist=fraud_edges, 
                              edge_color=self.colors['fraud_edge'], 
                              width=self.edge_width*2, alpha=0.8, ax=self.ax)
        
        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, nodelist=source_nodes, 
                              node_color=self.colors['source_node'], 
                              node_size=self.node_size, alpha=0.8, ax=self.ax)
        nx.draw_networkx_nodes(graph, pos, nodelist=target_nodes, 
                              node_color=self.colors['target_node'], 
                              node_size=self.node_size, alpha=0.8, ax=self.ax)
        
        # Highlight selected nodes and edges
        if self.highlighted_nodes:
            nx.draw_networkx_nodes(graph, pos, nodelist=list(self.highlighted_nodes), 
                                  node_color=self.colors['highlight'], 
                                  node_size=self.node_size*1.5, alpha=0.9, ax=self.ax)
        
        if self.highlighted_edges:
            nx.draw_networkx_edges(graph, pos, edgelist=list(self.highlighted_edges), 
                                  edge_color=self.colors['highlight'], 
                                  width=self.edge_width*3, alpha=0.9, ax=self.ax)
        
        # Add node labels for highlighted nodes
        if self.highlighted_nodes:
            labels = {n: str(n) for n in self.highlighted_nodes}
            nx.draw_networkx_labels(graph, pos, labels, font_size=self.font_size, ax=self.ax)
        
        self.ax.set_title('GNN Graph Structure (Source → Target)', fontsize=14)
        self.ax.axis('off')
    
    def _add_controls(self, graph, pos):
        """Add interactive controls to the visualization"""
        # Create control panel
        ax_controls = plt.axes([0.02, 0.02, 0.25, 0.3])
        ax_controls.axis('off')
        
        # Add buttons
        ax_highlight_fraud = plt.axes([0.02, 0.25, 0.12, 0.04])
        ax_highlight_normal = plt.axes([0.15, 0.25, 0.12, 0.04])
        ax_clear_highlight = plt.axes([0.02, 0.20, 0.12, 0.04])
        ax_show_stats = plt.axes([0.15, 0.20, 0.12, 0.04])
        
        # Create buttons
        btn_highlight_fraud = Button(ax_highlight_fraud, 'Highlight Fraud')
        btn_highlight_normal = Button(ax_highlight_normal, 'Highlight Normal')
        btn_clear_highlight = Button(ax_clear_highlight, 'Clear Highlight')
        btn_show_stats = Button(ax_show_stats, 'Show Stats')
        
        # Connect button events
        btn_highlight_fraud.on_clicked(lambda event: self._highlight_fraud_edges(graph))
        btn_highlight_normal.on_clicked(lambda event: self._highlight_normal_edges(graph))
        btn_clear_highlight.on_clicked(lambda event: self._clear_highlights(graph, pos))
        btn_show_stats.on_clicked(lambda event: self._show_graph_statistics(graph))
        
        # Add sliders
        ax_node_size = plt.axes([0.02, 0.15, 0.25, 0.02])
        ax_edge_width = plt.axes([0.02, 0.12, 0.25, 0.02])
        
        slider_node_size = Slider(ax_node_size, 'Node Size', 100, 800, valinit=self.node_size)
        slider_edge_width = Slider(ax_edge_width, 'Edge Width', 0.5, 3.0, valinit=self.edge_width)
        
        # Connect slider events
        slider_node_size.on_changed(lambda val: self._update_node_size(val, graph, pos))
        slider_edge_width.on_changed(lambda val: self._update_edge_width(val, graph, pos))
    
    def _add_info_panel(self, sample_data):
        """Add information panel showing graph statistics"""
        ax_info = plt.axes([0.75, 0.02, 0.23, 0.4])
        ax_info.axis('off')
        
        # Calculate statistics
        num_nodes = sample_data.num_nodes
        num_edges = sample_data.num_edges
        fraud_rate = sample_data.edge_labels.float().mean().item()
        num_source = len([n for n in range(num_nodes) if n < len(self.df['cc_num'].unique())])
        num_target = num_nodes - num_source
        
        # Create info text
        info_text = f"""
Graph Statistics:
• Total Nodes: {num_nodes}
• Source Nodes: {num_source}
• Target Nodes: {num_target}
• Total Edges: {num_edges}
• Fraud Rate: {fraud_rate:.3f}

Node Features:
• Source: {len(self.source_node_features)} features
• Target: {len(self.target_node_features)} features

Edge Features:
• {len(self.edge_features)} features

Sample Size: {len(self.df)} transactions
        """
        
        ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    def _highlight_fraud_edges(self, graph):
        """Highlight fraud edges in the graph"""
        fraud_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['is_fraud'] == 1]
        self.highlighted_edges = set(fraud_edges)
        
        # Get nodes connected to fraud edges
        fraud_nodes = set()
        for u, v in fraud_edges:
            fraud_nodes.add(u)
            fraud_nodes.add(v)
        self.highlighted_nodes = fraud_nodes
        
        self._redraw_graph(graph)
        print(f"Highlighted {len(fraud_edges)} fraud edges and {len(fraud_nodes)} connected nodes")
    
    def _highlight_normal_edges(self, graph):
        """Highlight normal edges in the graph"""
        normal_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['is_fraud'] == 0]
        self.highlighted_edges = set(normal_edges)
        
        # Get nodes connected to normal edges
        normal_nodes = set()
        for u, v in normal_edges:
            normal_nodes.add(u)
            normal_nodes.add(v)
        self.highlighted_nodes = normal_nodes
        
        self._redraw_graph(graph)
        print(f"Highlighted {len(normal_edges)} normal edges and {len(normal_nodes)} connected nodes")
    
    def _clear_highlights(self, graph, pos):
        """Clear all highlights"""
        self.highlighted_nodes.clear()
        self.highlighted_edges.clear()
        self._redraw_graph(graph)
        print("Cleared all highlights")
    
    def _show_graph_statistics(self, graph):
        """Show detailed graph statistics"""
        print("\n" + "="*50)
        print("DETAILED GRAPH STATISTICS")
        print("="*50)
        
        # Node statistics
        source_nodes = [n for n, d in graph.nodes(data=True) if d['type'] == 'source']
        target_nodes = [n for n, d in graph.nodes(data=True) if d['type'] == 'target']
        
        print(f"Nodes:")
        print(f"  Total: {graph.number_of_nodes()}")
        print(f"  Source: {len(source_nodes)}")
        print(f"  Target: {len(target_nodes)}")
        
        # Edge statistics
        fraud_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['is_fraud'] == 1]
        normal_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['is_fraud'] == 0]
        
        print(f"Edges:")
        print(f"  Total: {graph.number_of_edges()}")
        print(f"  Fraud: {len(fraud_edges)} ({len(fraud_edges)/graph.number_of_edges():.3f})")
        print(f"  Normal: {len(normal_edges)} ({len(normal_edges)/graph.number_of_edges():.3f})")
        
        # Feature statistics
        if graph.number_of_nodes() > 0:
            sample_node = list(graph.nodes(data=True))[0]
            node_features = sample_node[1]['features']
            print(f"Node Features: {len(node_features)} dimensions")
            print(f"  Mean: {np.mean(node_features):.4f}")
            print(f"  Std: {np.std(node_features):.4f}")
            print(f"  Min: {np.min(node_features):.4f}")
            print(f"  Max: {np.max(node_features):.4f}")
        
        if graph.number_of_edges() > 0:
            sample_edge = list(graph.edges(data=True))[0]
            edge_features = sample_edge[2]['features']
            print(f"Edge Features: {len(edge_features)} dimensions")
            print(f"  Mean: {np.mean(edge_features):.4f}")
            print(f"  Std: {np.std(edge_features):.4f}")
            print(f"  Min: {np.min(edge_features):.4f}")
            print(f"  Max: {np.max(edge_features):.4f}")
        
        # Degree statistics
        in_degrees = [d for n, d in graph.in_degree()]
        out_degrees = [d for n, d in graph.out_degree()]
        
        print(f"Degree Statistics:")
        print(f"  In-degree - Mean: {np.mean(in_degrees):.2f}, Max: {np.max(in_degrees)}")
        print(f"  Out-degree - Mean: {np.mean(out_degrees):.2f}, Max: {np.max(out_degrees)}")
    
    def _update_node_size(self, val, graph, pos):
        """Update node size in visualization"""
        self.node_size = val
        self._redraw_graph(graph)
    
    def _update_edge_width(self, val, graph, pos):
        """Update edge width in visualization"""
        self.edge_width = val
        self._redraw_graph(graph)
    
    def _redraw_graph(self, graph):
        """Redraw the graph with current settings"""
        if self.ax is not None:
            pos = nx.spring_layout(graph, k=1, iterations=50, seed=self.random_state)
            self._draw_graph(graph, pos)
            plt.draw()
    
    def explore_node_features(self, node_id):
        """Explore features of a specific node"""
        if self.graph_data is None:
            print("No graph data loaded")
            return
        
        if node_id >= self.graph_data.num_nodes:
            print(f"Node {node_id} does not exist")
            return
        
        print(f"\nNode {node_id} Features:")
        print("="*30)
        
        node_features = self.graph_data.x[node_id].cpu().numpy()
        num_source_nodes = len(self.df['cc_num'].unique())
        
        if node_id < num_source_nodes:
            print(f"Type: Source Node (Customer)")
            print(f"Features: {len(self.source_node_features)} dimensions")
            
            # Show feature names and values
            for i, feature_name in enumerate(self.source_node_features):
                if i < len(node_features):
                    print(f"  {feature_name}: {node_features[i]:.4f}")
        else:
            print(f"Type: Target Node (Merchant)")
            print(f"Features: {len(self.target_node_features)} dimensions")
            
            # Show feature names and values
            for i, feature_name in enumerate(self.target_node_features):
                if i < len(node_features):
                    print(f"  {feature_name}: {node_features[i]:.4f}")
        
        # Show all feature values
        print(f"\nAll feature values:")
        for i, val in enumerate(node_features):
            print(f"  Feature {i}: {val:.4f}")
    
    def explore_edge_features(self, edge_id):
        """Explore features of a specific edge"""
        if self.graph_data is None:
            print("No graph data loaded")
            return
        
        if edge_id >= self.graph_data.num_edges:
            print(f"Edge {edge_id} does not exist")
            return
        
        print(f"\nEdge {edge_id} Features:")
        print("="*30)
        
        edge_index = self.graph_data.edge_index.cpu().numpy()
        edge_attr = self.graph_data.edge_attr.cpu().numpy()
        edge_labels = self.graph_data.edge_labels.cpu().numpy()
        
        source = edge_index[0, edge_id]
        target = edge_index[1, edge_id]
        edge_features = edge_attr[edge_id]
        is_fraud = edge_labels[edge_id]
        
        print(f"Source Node: {source}")
        print(f"Target Node: {target}")
        print(f"Is Fraud: {is_fraud}")
        print(f"Features: {len(self.edge_features)} dimensions")
        
        # Show feature names and values
        for i, feature_name in enumerate(self.edge_features):
            if i < len(edge_features):
                print(f"  {feature_name}: {edge_features[i]:.4f}")
        
        # Show all feature values
        print(f"\nAll feature values:")
        for i, val in enumerate(edge_features):
            print(f"  Feature {i}: {val:.4f}")
    
    def show_sample_inference_data(self, num_samples=5):
        """Show sample data used for inference"""
        print(f"\nSample Inference Data ({num_samples} samples):")
        print("="*50)
        
        if self.graph_data is None:
            print("No graph data loaded")
            return
        
        # Sample random edges for inference
        sample_indices = np.random.choice(
            self.graph_data.num_edges, 
            size=min(num_samples, self.graph_data.num_edges), 
            replace=False
        )
        
        edge_index = self.graph_data.edge_index.cpu().numpy()
        edge_attr = self.graph_data.edge_attr.cpu().numpy()
        edge_labels = self.graph_data.edge_labels.cpu().numpy()
        
        for i, edge_id in enumerate(sample_indices):
            source = edge_index[0, edge_id]
            target = edge_index[1, edge_id]
            edge_features = edge_attr[edge_id]
            is_fraud = edge_labels[edge_id]
            
            print(f"\nSample {i+1} (Edge {edge_id}):")
            print(f"  Source Node: {source}")
            print(f"  Target Node: {target}")
            print(f"  Is Fraud: {is_fraud}")
            print(f"  Edge Features: {edge_features}")
            
            # Show node features
            source_features = self.graph_data.x[source].cpu().numpy()
            target_features = self.graph_data.x[target].cpu().numpy()
            
            print(f"  Source Node Features: {source_features}")
            print(f"  Target Node Features: {target_features}")
    
    def run_exploration(self):
        """Run the complete graph exploration"""
        print("Starting Graph Exploration")
        print("="*50)
        
        # Load data and build graph
        self.load_data_and_build_graph()
        
        # Create interactive visualization
        sample_nx, pos = self.create_interactive_visualization()
        
        # Show sample inference data
        self.show_sample_inference_data()
        
        print("\nGraph exploration completed!")
        print("Use the interactive controls to explore the graph structure.")
        
        return sample_nx, pos

def main():
    """Main function to run the graph exploration"""
    explorer = GraphExplorer(random_state=42)
    sample_nx, pos = explorer.run_exploration()
    
    return sample_nx, pos

if __name__ == '__main__':
    main() 