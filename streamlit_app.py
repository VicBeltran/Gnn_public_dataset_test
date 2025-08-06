import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import torch
import json
import os
from datetime import datetime
import pickle

from run_comparison import ModelComparisonPipeline
from gnn_wrapper import GNNTrainingWrapper
from training_pipeline import GenericTrainingPipeline
from models import ImprovedEdgeGraphSAGE
from visualization import ModelEvaluationVisualizer

# Page configuration
st.set_page_config(
    page_title="Graph Neural Network vs XGBoost Comparison",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .graph-container {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class GraphVisualizer:
    """Handles graph visualization and interaction"""
    
    def __init__(self):
        self.graph_data = None
        self.node_data = None
        self.edge_data = None
        
    def create_network_graph(self, data, selected_edge_idx=None):
        """Create an interactive network graph using Plotly"""
        # Convert PyTorch Geometric data to NetworkX
        edge_index = data.edge_index.cpu().numpy()
        edge_attr = data.edge_attr.cpu().numpy()
        node_features = data.x.cpu().numpy()
        edge_labels = data.edge_labels.cpu().numpy()
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes
        for i in range(data.num_nodes):
            G.add_node(i, features=node_features[i])
        
        # Add edges
        for i, (source, target) in enumerate(edge_index.T):
            G.add_edge(source, target, 
                      features=edge_attr[i], 
                      label=edge_labels[i],
                      edge_idx=i)
        
        # Get layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node text
            features = G.nodes[node]['features']
            node_text.append(f"Node {node}<br>Features: {features[:3]}...")
            
            # Color based on node type (source vs target)
            if node < len(G.nodes()) // 2:
                node_color.append('lightblue')  # Source nodes
            else:
                node_color.append('lightgreen')  # Target nodes
        
        # Prepare edge traces
        edge_x = []
        edge_y = []
        edge_text = []
        edge_color = []
        
        for source, target, data in G.edges(data=True):
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Edge text
            features = data['features']
            label = data['label']
            edge_idx = data['edge_idx']
            edge_text.append(f"Edge {edge_idx}<br>Label: {label}<br>Features: {features[:3]}...")
            
            # Color based on fraud label
            if label == 1:
                edge_color.append('red')
            else:
                edge_color.append('gray')
        
        # Create the plot
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='gray'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=10,
                color=node_color,
                line=dict(width=2, color='white')
            ),
            showlegend=False
        ))
        
        # Highlight selected edge if provided
        if selected_edge_idx is not None:
            for source, target, data in G.edges(data=True):
                if data['edge_idx'] == selected_edge_idx:
                    x0, y0 = pos[source]
                    x1, y1 = pos[target]
                    fig.add_trace(go.Scatter(
                        x=[x0, x1], y=[y0, y1],
                        line=dict(width=3, color='yellow'),
                        mode='lines',
                        showlegend=False
                    ))
        
        fig.update_layout(
            title="Graph Visualization",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        return fig
    
    def create_subgraph_visualization(self, data, center_edge_idx, num_hops=2):
        """Create visualization of a subgraph around a specific edge"""
        # Get the edge
        edge_index = data.edge_index.cpu().numpy()
        source, target = edge_index[:, center_edge_idx]
        
        # Get k-hop subgraph
        gnn_wrapper = GNNTrainingWrapper()
        subgraph_data = gnn_wrapper.sample_subgraph_around_edges(
            data, [center_edge_idx], num_hops, max_nodes=50
        )
        
        if subgraph_data is None:
            st.error("Could not create subgraph")
            return None
        
        # Create visualization
        fig = self.create_network_graph(subgraph_data, center_edge_idx)
        fig.update_layout(title=f"Subgraph around Edge {center_edge_idx} ({num_hops}-hop)")
        
        return fig

class ModelInference:
    """Handles model loading and inference"""
    
    def __init__(self):
        self.xgb_model = None
        self.gnn_model = None
        self.models_loaded = False
        
    def load_models(self, model_dir="./models"):
        """Load trained models"""
        try:
            # Load XGBoost model
            xgb_files = [f for f in os.listdir(f"{model_dir}/xgb_models") if f.endswith('.pkl')]
            if xgb_files:
                latest_xgb = max(xgb_files, key=lambda x: os.path.getctime(f"{model_dir}/xgb_models/{x}"))
                with open(f"{model_dir}/xgb_models/{latest_xgb}", 'rb') as f:
                    self.xgb_model = pickle.load(f)
                st.success(f"Loaded XGBoost model: {latest_xgb}")
            
            # Load GNN model
            gnn_files = [f for f in os.listdir(f"{model_dir}/gnn_models") if f.endswith('.pt')]
            if gnn_files:
                latest_gnn = max(gnn_files, key=lambda x: os.path.getctime(f"{model_dir}/gnn_models/{x}"))
                # Load GNN model (this would need the model architecture)
                st.success(f"Found GNN model: {latest_gnn}")
            
            self.models_loaded = True
            
        except Exception as e:
            st.error(f"Error loading models: {e}")
    
    def predict_edge(self, edge_features, model_type='xgb'):
        """Predict fraud probability for a specific edge"""
        if not self.models_loaded:
            return None, "Models not loaded"
        
        try:
            if model_type == 'xgb' and self.xgb_model is not None:
                # Reshape features for XGBoost
                features = np.array(edge_features).reshape(1, -1)
                prediction = self.xgb_model.predict_proba(features)[0]
                return prediction[1], "XGBoost prediction"
            else:
                return None, "Model not available"
        except Exception as e:
            return None, f"Prediction error: {e}"

def main():
    st.markdown('<h1 class="main-header">🔍 Graph Neural Network vs XGBoost Comparison</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Data Overview", "Graph Visualization", "Subgraph Analysis", "Model Inference", "Model Comparison"]
    )
    
    # Initialize components
    if 'graph_visualizer' not in st.session_state:
        st.session_state.graph_visualizer = GraphVisualizer()
    if 'model_inference' not in st.session_state:
        st.session_state.model_inference = ModelInference()
    
    # Load data
    if 'data' not in st.session_state:
        with st.spinner("Loading example data..."):
            from run_comparison import create_example_data
            df = create_example_data()
            st.session_state.data = df
            st.session_state.graph_data = None
    
    if page == "Data Overview":
        show_data_overview()
    elif page == "Graph Visualization":
        show_graph_visualization()
    elif page == "Subgraph Analysis":
        show_subgraph_analysis()
    elif page == "Model Inference":
        show_model_inference()
    elif page == "Model Comparison":
        show_model_comparison()

def show_data_overview():
    st.header("📊 Data Overview")
    
    df = st.session_state.data
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", len(df))
    
    with col2:
        st.metric("Fraud Rate", f"{df['is_fraud'].mean():.2%}")
    
    with col3:
        st.metric("Unique Source Nodes", df['source_node'].nunique())
    
    with col4:
        st.metric("Unique Target Nodes", df['target_node'].nunique())
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(df.head(10))
    
    # Feature distributions
    st.subheader("Feature Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x='amount', color='is_fraud', 
                          title="Amount Distribution by Fraud Status")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(df, x='hour', color='is_fraud',
                          title="Hour Distribution by Fraud Status")
        st.plotly_chart(fig, use_container_width=True)
    
    # Fraud patterns
    st.subheader("Fraud Patterns")
    
    fraud_df = df[df['is_fraud'] == 1]
    normal_df = df[df['is_fraud'] == 0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(fraud_df, y='amount', title="Fraudulent Transaction Amounts")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(normal_df, y='amount', title="Normal Transaction Amounts")
        st.plotly_chart(fig, use_container_width=True)

def show_graph_visualization():
    st.header("🕸️ Graph Visualization")
    
    df = st.session_state.data
    
    # Build graph if not already built
    if st.session_state.graph_data is None:
        with st.spinner("Building graph..."):
            gnn_wrapper = GNNTrainingWrapper()
            
            # Define features
            node_features = ['amount', 'hour', 'day_of_week', 'month', 'is_weekend']
            edge_features = ['amount', 'hour', 'day_of_week', 'month', 'is_weekend']
            
            # Build graph
            graph_data = gnn_wrapper.build_graph_from_tabular(
                df=df,
                node_features=node_features,
                edge_features=edge_features,
                target_col='is_fraud',
                source_node_col='source_node',
                target_node_col='target_node'
            )
            
            st.session_state.graph_data = graph_data
            st.session_state.gnn_wrapper = gnn_wrapper
    
    # Graph controls
    col1, col2 = st.columns(2)
    
    with col1:
        max_nodes = st.slider("Maximum nodes to display", 50, 500, 200)
    
    with col2:
        show_fraud_only = st.checkbox("Show fraud edges only", False)
    
    # Create visualization
    if st.session_state.graph_data is not None:
        graph_data = st.session_state.graph_data
        
        # Filter graph if needed
        if show_fraud_only:
            fraud_mask = graph_data.edge_labels == 1
            filtered_data = type(graph_data)(
                x=graph_data.x,
                edge_index=graph_data.edge_index[:, fraud_mask],
                edge_attr=graph_data.edge_attr[fraud_mask],
                edge_labels=graph_data.edge_labels[fraud_mask]
            )
        else:
            filtered_data = graph_data
        
        # Create visualization
        fig = st.session_state.graph_visualizer.create_network_graph(filtered_data)
        
        # Display the graph
        st.plotly_chart(fig, use_container_width=True)
        
        # Graph statistics
        st.subheader("Graph Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Nodes", filtered_data.num_nodes)
        
        with col2:
            st.metric("Total Edges", filtered_data.num_edges)
        
        with col3:
            fraud_rate = filtered_data.edge_labels.float().mean()
            st.metric("Fraud Rate", f"{fraud_rate:.2%}")

def show_subgraph_analysis():
    st.header("🔍 Subgraph Analysis")
    
    if st.session_state.graph_data is None:
        st.warning("Please build the graph first in the Graph Visualization page.")
        return
    
    graph_data = st.session_state.graph_data
    
    # Subgraph controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        edge_idx = st.number_input("Edge Index", 0, graph_data.num_edges-1, 0)
    
    with col2:
        num_hops = st.slider("Number of hops", 1, 3, 2)
    
    with col3:
        max_subgraph_size = st.slider("Max subgraph size", 20, 200, 50)
    
    # Get edge information
    edge_index = graph_data.edge_index.cpu().numpy()
    edge_attr = graph_data.edge_attr.cpu().numpy()
    edge_labels = graph_data.edge_labels.cpu().numpy()
    
    source, target = edge_index[:, edge_idx]
    edge_features = edge_attr[edge_idx]
    edge_label = edge_labels[edge_idx]
    
    # Display edge information
    st.subheader(f"Selected Edge {edge_idx}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Source Node", int(source))
    
    with col2:
        st.metric("Target Node", int(target))
    
    with col3:
        st.metric("Fraud Label", "Fraud" if edge_label == 1 else "Normal")
    
    # Edge features
    st.subheader("Edge Features")
    feature_names = ['amount', 'hour', 'day_of_week', 'month', 'is_weekend']
    
    for i, (name, value) in enumerate(zip(feature_names, edge_features)):
        st.metric(f"{name}", f"{value:.4f}")
    
    # Create subgraph visualization
    st.subheader("Subgraph Visualization")
    
    gnn_wrapper = st.session_state.gnn_wrapper
    subgraph_data = gnn_wrapper.sample_subgraph_around_edges(
        graph_data, [edge_idx], num_hops, max_subgraph_size
    )
    
    if subgraph_data is not None:
        fig = st.session_state.graph_visualizer.create_network_graph(subgraph_data, edge_idx)
        fig.update_layout(title=f"Subgraph around Edge {edge_idx} ({num_hops}-hop)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Subgraph statistics
        st.subheader("Subgraph Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Subgraph Nodes", subgraph_data.num_nodes)
        
        with col2:
            st.metric("Subgraph Edges", subgraph_data.num_edges)
        
        with col3:
            subgraph_fraud_rate = subgraph_data.edge_labels.float().mean()
            st.metric("Subgraph Fraud Rate", f"{subgraph_fraud_rate:.2%}")
    else:
        st.error("Could not create subgraph")

def show_model_inference():
    st.header("🤖 Model Inference")
    
    # Load models
    if not st.session_state.model_inference.models_loaded:
        with st.spinner("Loading models..."):
            st.session_state.model_inference.load_models()
    
    # Inference controls
    st.subheader("Edge Selection")
    
    if st.session_state.graph_data is None:
        st.warning("Please build the graph first in the Graph Visualization page.")
        return
    
    graph_data = st.session_state.graph_data
    
    # Select edge for inference
    edge_idx = st.number_input("Select Edge for Inference", 0, graph_data.num_edges-1, 0)
    
    # Get edge information
    edge_index = graph_data.edge_index.cpu().numpy()
    edge_attr = graph_data.edge_attr.cpu().numpy()
    edge_labels = graph_data.edge_labels.cpu().numpy()
    
    source, target = edge_index[:, edge_idx]
    edge_features = edge_attr[edge_idx]
    true_label = edge_labels[edge_idx]
    
    # Display edge information
    st.subheader("Edge Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Edge Index", edge_idx)
    
    with col2:
        st.metric("Source Node", int(source))
    
    with col3:
        st.metric("Target Node", int(target))
    
    with col4:
        st.metric("True Label", "Fraud" if true_label == 1 else "Normal")
    
    # Model selection
    st.subheader("Model Selection")
    
    model_type = st.selectbox("Choose Model", ["XGBoost", "GNN"])
    
    # Perform inference
    if st.button("Perform Inference"):
        with st.spinner("Running inference..."):
            prediction, message = st.session_state.model_inference.predict_edge(
                edge_features, model_type.lower()
            )
            
            if prediction is not None:
                st.success(f"Prediction: {prediction:.4f}")
                st.info(f"Message: {message}")
                
                # Display prediction vs true label
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Predicted Probability", f"{prediction:.4f}")
                
                with col2:
                    st.metric("True Label", "Fraud" if true_label == 1 else "Normal")
                
                # Prediction interpretation
                threshold = 0.5
                predicted_label = 1 if prediction > threshold else 0
                
                if predicted_label == true_label:
                    st.success("✅ Prediction Correct!")
                else:
                    st.error("❌ Prediction Incorrect!")
            else:
                st.error(f"Inference failed: {message}")

def show_model_comparison():
    st.header("📈 Model Comparison")
    
    # Check if results exist
    results_dir = "./outputs"
    if not os.path.exists(results_dir):
        st.warning("No results found. Please run the training pipeline first.")
        return
    
    # Load results
    result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    
    if not result_files:
        st.warning("No result files found.")
        return
    
    # Select result file
    selected_file = st.selectbox("Select Result File", result_files)
    
    if selected_file:
        with open(os.path.join(results_dir, selected_file), 'r') as f:
            results = json.load(f)
        
        # Display results
        st.subheader("Model Performance")
        
        if 'comparison' in results:
            comparison = results['comparison']
            
            # Create comparison table
            df_comparison = pd.DataFrame({
                'Metric': comparison['metric'],
                'XGBoost': comparison['XGBoost'],
                'GNN': comparison['GNN']
            })
            
            st.dataframe(df_comparison, use_container_width=True)
            
            # Performance visualization
            st.subheader("Performance Comparison")
            
            fig = px.bar(df_comparison, x='Metric', y=['XGBoost', 'GNN'],
                        title="Model Performance Comparison",
                        barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed results
        if 'xgb_results' in results and 'gnn_results' in results:
            st.subheader("Detailed Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**XGBoost Results**")
                xgb_results = results['xgb_results']
                for key, value in xgb_results.items():
                    if isinstance(value, (int, float)):
                        st.metric(key, f"{value:.4f}")
            
            with col2:
                st.write("**GNN Results**")
                gnn_results = results['gnn_results']
                for key, value in gnn_results.items():
                    if isinstance(value, (int, float)):
                        st.metric(key, f"{value:.4f}")

if __name__ == "__main__":
    main() 