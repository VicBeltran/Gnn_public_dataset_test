"""
Streamlit GNN Inference Visualization App
Interactive interface for exploring GNN inference with subgraph visualization
"""

import streamlit as st
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
import warnings
import os

# Import our custom modules
from graph_debug_explorer import GraphDebugExplorer
from models import ImprovedEdgeGraphSAGE, HeterogeneousEdgeGraphSAGE
from gnn_wrapper import GNNTrainingWrapper

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Explorador de Inferência GNN",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .highlight-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_graph_data():
    """Load and cache the graph data"""
    
    try:
        explorer = GraphDebugExplorer(random_state=42)
        graph_data = explorer.load_data_and_build_graph()
        return explorer, graph_data
    except Exception as e:
        st.error(f"Error loading graph data: {e}")
        return None, None

def get_edge_range(graph_data):
    """Get the range of available edge indices"""
    if graph_data is None:
        return 0, 0
    
    return 0, graph_data.num_edges - 1

def create_interactive_graph_plot(subgraph_data, target_edge_indices, source_node_features, target_node_features, edge_features):
    """Create an interactive Plotly graph visualization"""
    
    if subgraph_data is None:
        return None
    
    # Convert to numpy for easier manipulation
    edge_index = subgraph_data.edge_index.cpu().numpy()
    edge_attr = subgraph_data.edge_attr.cpu().numpy()
    edge_labels = subgraph_data.edge_labels.cpu().numpy()
    node_features = subgraph_data.x.cpu().numpy()
    
    # Determine node types (source vs target)
    num_source_nodes = len(np.unique(edge_index[0]))  # Approximate
    node_types = []
    for i in range(subgraph_data.num_nodes):
        if i < num_source_nodes:
            node_types.append('Cliente')
        else:
            node_types.append('Comerciante')
    
    # Create node positions using spring layout simulation
    import networkx as nx
    G = nx.DiGraph()
    for i in range(subgraph_data.num_nodes):
        G.add_node(i)
    for i in range(subgraph_data.num_edges):
        G.add_edge(edge_index[0, i], edge_index[1, i])
    
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
    # Extract positions
    node_x = [pos[node][0] for node in range(subgraph_data.num_nodes)]
    node_y = [pos[node][1] for node in range(subgraph_data.num_nodes)]
    
    # Create edge traces
    edge_traces = []
    
    # Normal edges (non-fraud)
    normal_edges = []
    normal_x = []
    normal_y = []
    for i in range(subgraph_data.num_edges):
        if edge_labels[i] == 0:  # Normal transaction
            source = edge_index[0, i]
            target = edge_index[1, i]
            normal_x.extend([node_x[source], node_x[target], None])
            normal_y.extend([node_y[source], node_y[target], None])
            normal_edges.append(i)
    
    if normal_edges:
        # Create hover text for normal edges
        normal_hover_text = []
        for i in normal_edges:
            source = edge_index[0, i]
            target = edge_index[1, i]
            # Get edge features for hover text
            edge_feat = edge_attr[i]
            feat_text = ""
            try:
                if len(edge_features) > 0 and len(edge_feat) > 0:
                    feat_text = "<br>Features: " + ", ".join([f"{feat[:10]}: {val:.3f}" for feat, val in zip(edge_features[:3], edge_feat[:3])])
            except Exception:
                feat_text = ""  # Fallback if feature extraction fails
            # Add hover text for each point in the line segment
            normal_hover_text.extend([
                f"Aresta {i}<br>Origem: {source}<br>Destino: {target}<br>Rótulo: Normal (0){feat_text}",
                f"Aresta {i}<br>Origem: {source}<br>Destino: {target}<br>Rótulo: Normal (0){feat_text}",
                None  # None value for line breaks
            ])
        
        edge_traces.append(go.Scatter(
            x=normal_x, y=normal_y,
            line=dict(width=1, color='#888'),
            hoverinfo='text',
            hovertext=normal_hover_text,
            mode='lines',
            name='Transações Normais',
            showlegend=True
        ))
    
    # Fraud edges
    fraud_edges = []
    fraud_x = []
    fraud_y = []
    for i in range(subgraph_data.num_edges):
        if edge_labels[i] == 1:  # Fraud transaction
            source = edge_index[0, i]
            target = edge_index[1, i]
            fraud_x.extend([node_x[source], node_x[target], None])
            fraud_y.extend([node_y[source], node_y[target], None])
            fraud_edges.append(i)
    
    if fraud_edges:
        # Create hover text for fraud edges
        fraud_hover_text = []
        for i in fraud_edges:
            source = edge_index[0, i]
            target = edge_index[1, i]
            # Get edge features for hover text
            edge_feat = edge_attr[i]
            feat_text = ""
            if len(edge_features) > 0 and len(edge_feat) > 0:
                feat_text = "<br>Features: " + ", ".join([f"{feat[:10]}: {val:.3f}" for feat, val in zip(edge_features[:3], edge_feat[:3])])
            # Add hover text for each point in the line segment
            fraud_hover_text.extend([
                f"Aresta {i}<br>Origem: {source}<br>Destino: {target}<br>Rótulo: Fraudulenta (1){feat_text}",
                f"Aresta {i}<br>Origem: {source}<br>Destino: {target}<br>Rótulo: Fraudulenta (1){feat_text}",
                None  # None value for line breaks
            ])
        
        edge_traces.append(go.Scatter(
            x=fraud_x, y=fraud_y,
            line=dict(width=3, color='red'),
            hoverinfo='text',
            hovertext=fraud_hover_text,
            mode='lines',
            name='Transações Fraudulentas',
            showlegend=True
        ))
    
    # Highlight target edges
    if target_edge_indices is not None:
        target_x = []
        target_y = []
        for edge_idx in target_edge_indices:
            # Find the edge in the subgraph
            if hasattr(subgraph_data, 'original_edge_indices'):
                # Map back to subgraph edge index
                original_indices = subgraph_data.original_edge_indices.cpu().numpy()
                subgraph_edge_idx = np.where(original_indices == edge_idx)[0]
                if len(subgraph_edge_idx) > 0:
                    edge_idx_in_subgraph = subgraph_edge_idx[0]
                    source = edge_index[0, edge_idx_in_subgraph]
                    target = edge_index[1, edge_idx_in_subgraph]
                    target_x.extend([node_x[source], node_x[target], None])
                    target_y.extend([node_y[source], node_y[target], None])
        
        if target_x:  # Only add trace if we have target edges
            # Create hover text for target edges
            target_hover_text = []
            for edge_idx in target_edge_indices:
                if hasattr(subgraph_data, 'original_edge_indices'):
                    original_indices = subgraph_data.original_edge_indices.cpu().numpy()
                    subgraph_edge_idx = np.where(original_indices == edge_idx)[0]
                    if len(subgraph_edge_idx) > 0:
                        edge_idx_in_subgraph = subgraph_edge_idx[0]
                        source = edge_index[0, edge_idx_in_subgraph]
                        target = edge_index[1, edge_idx_in_subgraph]
                        # Get edge features for hover text
                        edge_feat = edge_attr[edge_idx_in_subgraph]
                        feat_text = ""
                        if len(edge_features) > 0 and len(edge_feat) > 0:
                            feat_text = "<br>Features: " + ", ".join([f"{feat[:10]}: {val:.3f}" for feat, val in zip(edge_features[:3], edge_feat[:3])])
                        # Get the actual label for the selected edge
                        actual_label = edge_labels[edge_idx_in_subgraph]
                        label_text = "Fraudulenta (1)" if actual_label == 1 else "Normal (0)"
                        # Add hover text for each point in the line segment
                        target_hover_text.extend([
                            f"Aresta {edge_idx}<br>Origem: {source}<br>Destino: {target}<br>Rótulo: {label_text}<br>Tipo: Selecionada{feat_text}",
                            f"Aresta {edge_idx}<br>Origem: {source}<br>Destino: {target}<br>Rótulo: {label_text}<br>Tipo: Selecionada{feat_text}",
                            None  # None value for line breaks
                        ])
            
            edge_traces.append(go.Scatter(
                x=target_x, y=target_y,
                line=dict(width=5, color='yellow'),
                hoverinfo='text',
                hovertext=target_hover_text,
                mode='lines',
                name='Aresta Selecionada',
                showlegend=True
            ))
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[f"Nó {i}<br>Tipo: {node_types[i]}" for i in range(subgraph_data.num_nodes)],
        textposition="top center",
        textfont=dict(size=8),
        marker=dict(
            size=15,
            color=[1 if node_types[i] == 'Customer' else 2 for i in range(subgraph_data.num_nodes)],
            colorscale='Viridis',
            line=dict(width=2, color='white'),
            showscale=False
        ),
        name='Nós'
    )
    
    # Create the figure
    fig = go.Figure(data=edge_traces + [node_trace])
    
    fig.update_layout(
        title=dict(
            text=f'Subgrafo de Inferência GNN ({subgraph_data.num_nodes} nós, {subgraph_data.num_edges} arestas)',
            font=dict(size=16)
        ),
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20,l=5,r=80,t=40),
        legend=dict(
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.1)',
            bordercolor='rgba(0,0,0,0.3)',
            borderwidth=1
        ),
        annotations=[ dict(
            text="Amarelo: Aresta Selecionada | Vermelho: Fraude | Cinza: Normal",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002 ) ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=800,
        height=600
    )
    
    return fig

def get_node_properties(node_idx, subgraph_data, source_node_features, target_node_features):
    """Get properties for a specific node"""
    if subgraph_data is None or node_idx >= subgraph_data.num_nodes:
        return None
    
    node_features = subgraph_data.x[node_idx].cpu().numpy()
    num_source_nodes = len(np.unique(subgraph_data.edge_index[0].cpu().numpy()))
    
    if node_idx < num_source_nodes:
        # Customer node
        properties = {
            'node_id': node_idx,
            'type': 'Customer',
            'features': dict(zip(source_node_features, node_features))
        }
    else:
        # Merchant node
        properties = {
            'node_id': node_idx,
            'type': 'Merchant',
            'features': dict(zip(target_node_features, node_features))
        }
    
    return properties

def get_edge_properties(edge_idx, subgraph_data, edge_features):
    """Get properties for a specific edge"""
    if subgraph_data is None or edge_idx >= subgraph_data.num_edges:
        return None
    
    edge_index = subgraph_data.edge_index.cpu().numpy()
    edge_attr = subgraph_data.edge_attr.cpu().numpy()
    edge_labels = subgraph_data.edge_labels.cpu().numpy()
    
    source = edge_index[0, edge_idx]
    target = edge_index[1, edge_idx]
    features = edge_attr[edge_idx]
    is_fraud = edge_labels[edge_idx]
    
    properties = {
        'edge_id': edge_idx,
        'source_node': source,
        'target_node': target,
        'is_fraud': bool(is_fraud),
        'features': dict(zip(edge_features, features))
    }
    
    return properties

def perform_inference_on_edge(explorer, graph_data, target_edge_idx):
    """Perform inference on a specific edge using the trained HeterogeneousEdgeGraphSAGE model"""
    try:
        # Load the trained model
        model_path = "models/gnn_models/gnn_model_20250806_152252.pt"
        params_path = "models/gnn_models/gnn_params_20250806_152252.json"
        
        if not os.path.exists(model_path):
            st.error(f"Modelo treinado não encontrado em {model_path}")
            return None, "Arquivo do modelo não encontrado"
        
        # Load model parameters
        import json
        with open(params_path, 'r') as f:
            model_params = json.load(f)
        
        # Load the state dict to determine model dimensions
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Try to determine dimensions from the state dict
        source_feature_dim = None
        target_feature_dim = None
        edge_dim = None
        
        for key in state_dict.keys():
            if 'source_encoder.0.weight' in key:
                source_feature_dim = state_dict[key].shape[1]
            elif 'target_encoder.0.weight' in key:
                target_feature_dim = state_dict[key].shape[1]
            elif 'edge_mlp.0.weight' in key:
                # The edge_mlp.0.weight has shape [hidden_channels, edge_dim]
                # So edge_dim is the second dimension
                edge_dim = state_dict[key].shape[1]
        
        # Use defaults if not found
        if source_feature_dim is None:
            source_feature_dim = 11  # 9 original + 2 additional
        if target_feature_dim is None:
            target_feature_dim = 4   # 2 original + 2 additional
        if edge_dim is None:
            edge_dim = 7  # Default for edge features
        
        # Check if the edge dimension matches the current data
        current_edge_dim = graph_data.edge_attr.shape[1]
        if edge_dim != current_edge_dim:
            print(f"Edge dimension mismatch: Model expects {edge_dim}, but data has {current_edge_dim}")
            print("This suggests the model was trained with different edge features.")
            print("Using the current data's edge dimension for inference.")
            edge_dim = current_edge_dim
        
        # Ensure we have the correct edge dimension
        if edge_dim is None:
            st.error("Não foi possível determinar a dimensão da aresta")
            return None, "Não foi possível determinar a dimensão da aresta"
        
        print(f"Model dimensions: Source={source_feature_dim}, Target={target_feature_dim}, Edge={edge_dim}")
        
        # Initialize the model with the determined parameters
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
        
        # Configure the GNN wrapper for inference (same as in evaluate_models.py)
        explorer.gnn_wrapper.max_subgraph_size = 500  # Max nodes in subgraph
        explorer.gnn_wrapper.num_hops = 2  # 2-hop neighborhood for realistic inference
        
        # Store the feature dimensions in the wrapper for heterogeneous inference
        explorer.gnn_wrapper.source_feature_dim = source_feature_dim
        explorer.gnn_wrapper.target_feature_dim = target_feature_dim
        
        # Initialize heterogeneous features in the wrapper (required for _create_heterogeneous_data)
        # Get the source and target features from the GNN wrapper that built the graph
        if hasattr(explorer.gnn_wrapper, 'source_features') and hasattr(explorer.gnn_wrapper, 'target_features'):
            # The features are already stored in the GNN wrapper from when the graph was built
            print(f"Using stored source features: {explorer.gnn_wrapper.source_features.shape}")
            print(f"Using stored target features: {explorer.gnn_wrapper.target_features.shape}")
        else:
            # Fallback: create dummy features if not available
            print("Source and target features not found in GNN wrapper, using fallback")
            # Get the number of source and target nodes from the graph data
            edge_index = graph_data.edge_index.cpu().numpy()
            source_nodes = np.unique(edge_index[0])
            target_nodes = np.unique(edge_index[1])
            
            # Create dummy features with the correct dimensions
            explorer.gnn_wrapper.source_features = np.random.randn(len(source_nodes), source_feature_dim)
            explorer.gnn_wrapper.target_features = np.random.randn(len(target_nodes), target_feature_dim)
        
        print(f"Using subgraph sampling with {explorer.gnn_wrapper.num_hops} hops and max {explorer.gnn_wrapper.max_subgraph_size} nodes")
        
        # Use the proper inference method from evaluate_models.py
        with torch.no_grad():
            # Perform inference using subgraph sampling
            predictions, probabilities = explorer.gnn_wrapper.inference_with_subgraph_sampling(
                model, graph_data, [target_edge_idx], batch_size=1
            )
            
            if len(probabilities) > 0:
                fraud_probability = probabilities[0]  # Get the prediction for our target edge
                
                # Get the actual label for comparison
                edge_labels = graph_data.edge_labels.cpu().numpy()
                if target_edge_idx < len(edge_labels):
                    actual_label = edge_labels[target_edge_idx]
                    actual_text = "Fraudulenta" if actual_label == 1 else "Normal"
                    
                    return fraud_probability, f"Previsão GNN real para aresta {target_edge_idx} (Real: {actual_text})"
                else:
                    return fraud_probability, f"Previsão GNN real para aresta {target_edge_idx}"
            else:
                return None, "Nenhuma previsão retornada pelo modelo"
        
    except Exception as e:
        st.error(f"Erro durante a inferência: {str(e)}")
        return None, f"Erro durante a inferência: {str(e)}"

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Explorador de Inferência GNN</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Carregando dados do grafo..."):
        explorer, graph_data = load_graph_data()
    
    if explorer is None or graph_data is None:
        st.error("Falha ao carregar dados do grafo. Verifique se os arquivos de dados existem.")
        st.stop()
        return
    
    # Show success message
    #st.success(f"✅ Graph loaded successfully! {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    # Sidebar for controls
    st.sidebar.header("🎛️ Controles")
    
    # Get edge range
    try:
        min_edge, max_edge = get_edge_range(graph_data)
        st.sidebar.markdown(f"**Índices de arestas disponíveis:** {min_edge} a {max_edge}")
        
        # Edge selection
        selected_edge_idx = st.sidebar.number_input(
            "Selecionar Índice da Aresta:",
            min_value=min_edge,
            max_value=max_edge,
            value=min_edge,
            step=1,
            help="Digite o índice da aresta que você quer explorar"
        )
    except Exception as e:
        st.error(f"Error getting edge range: {e}")
        st.stop()
        return
    
    # Get edge button
    if st.sidebar.button("🔍 Obter Aresta", type="primary"):
        st.session_state.selected_edge = selected_edge_idx
        st.session_state.show_graph = True
    
    # Model verification
    st.sidebar.markdown("---")
    st.sidebar.header("🤖 Status do Modelo")
    
    model_path = "models/gnn_models/gnn_model_20250806_152252.pt"
    params_path = "models/gnn_models/gnn_params_20250806_152252.json"
    
    if os.path.exists(model_path) and os.path.exists(params_path):
        st.sidebar.success("✅ Modelo treinado encontrado")
        try:
            import json
            with open(params_path, 'r') as f:
                model_params = json.load(f)
            st.sidebar.info(f"Tipo do modelo: HeterogeneousEdgeGraphSAGE")
            st.sidebar.info(f"Canais ocultos: {model_params.get('hidden_channels', 'N/A')}")
            st.sidebar.info(f"Dropout: {model_params.get('dropout', 'N/A')}")
            
            # Try to determine dimensions from state dict
            try:
                state_dict = torch.load(model_path, map_location='cpu')
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
                
                if source_feature_dim:
                    st.sidebar.info(f"Características da fonte: {source_feature_dim}")
                if target_feature_dim:
                    st.sidebar.info(f"Características do destino: {target_feature_dim}")
                # if edge_dim:
                #     st.sidebar.info(f"Características da aresta: {edge_dim}")
            except Exception as e:
                st.sidebar.warning(f"Não foi possível determinar as dimensões do modelo: {e}")
        except Exception as e:
            st.sidebar.error(f"Erro ao ler parâmetros do modelo: {e}")
    else:
        st.sidebar.error("❌ Modelo treinado não encontrado")
        st.sidebar.info("A inferência usará previsões simuladas")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Visualização do Grafo")
        
        if hasattr(st.session_state, 'show_graph') and st.session_state.show_graph:
            selected_edge_idx = st.session_state.selected_edge
            
            # Get subgraph around the selected edge
            with st.spinner("Amostrando subgrafo..."):
                subgraph_data = explorer.gnn_wrapper.sample_subgraph_around_edges(
                    graph_data, [selected_edge_idx], max_nodes=500
                )
            
            if subgraph_data is not None:
                # Store subgraph data in session state for later use
                st.session_state.subgraph_data = subgraph_data
                
                # Create interactive graph
                fig = create_interactive_graph_plot(
                    subgraph_data, 
                    [selected_edge_idx],
                    explorer.source_node_features,
                    explorer.target_node_features,
                    explorer.edge_features
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Subgraph statistics
                    st.markdown("### 📈 Estatísticas do Subgrafo")
                    col_stats1, col_stats2, col_stats3 = st.columns(3)
                    
                    with col_stats1:
                        st.metric("Nós", subgraph_data.num_nodes)
                    with col_stats2:
                        st.metric("Arestas", subgraph_data.num_edges)
                    with col_stats3:
                        fraud_rate = subgraph_data.edge_labels.float().mean().item()
                        st.metric("Taxa de Fraude", f"{fraud_rate:.3f}")
                    
                    # Inference button
                    if st.button("🚀 Executar Inferência", type="secondary"):
                        with st.spinner("Executando inferência..."):
                            prediction_score, message = perform_inference_on_edge(
                                explorer, graph_data, selected_edge_idx
                            )
                        
                        if prediction_score is not None:
                            st.markdown("### 🎯 Resultados da Inferência")
                            
                            # Create a gauge chart for the prediction score
                            fig_gauge = go.Figure(go.Indicator(
                                mode = "gauge+number+delta",
                                value = prediction_score,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Probabilidade de Fraude"},
                                delta = {'reference': 0.5},
                                gauge = {
                                    'axis': {'range': [None, 1]},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [0, 0.3], 'color': "lightgray"},
                                        {'range': [0.3, 0.7], 'color': "yellow"},
                                        {'range': [0.7, 1], 'color': "red"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 0.7
                                    }
                                }
                            ))
                            
                            fig_gauge.update_layout(height=300)
                            st.plotly_chart(fig_gauge, use_container_width=True)
                            
                            st.info(f"**Mensagem:** {message}")
                        else:
                            st.error(f"Inferência falhou: {message}")
                    
                    # Show some edge statistics
                    edge_labels_np = subgraph_data.edge_labels.cpu().numpy()
                    fraud_count = np.sum(edge_labels_np == 1)
                    normal_count = np.sum(edge_labels_np == 0)
                    st.markdown(f"**Distribuição de Arestas do Subgrafo:** {fraud_count} fraudulentas, {normal_count} normais")
                    
            else:
                st.error("Falha ao amostrar subgrafo ao redor da aresta selecionada.")
        else:
            st.info("👆 Selecione um índice de aresta e clique em 'Obter Aresta' para começar a explorar.")
    
    with col2:
        st.header("ℹ️ Painel de Informações")
        
        if hasattr(st.session_state, 'show_graph') and st.session_state.show_graph:
            selected_edge_idx = st.session_state.selected_edge
            
            # Show selected edge information
            st.markdown("### 📋 Aresta Selecionada")
            st.markdown(f"**Índice da Aresta:** {selected_edge_idx}")
            
            # Get edge properties from the full graph
            edge_index = graph_data.edge_index.cpu().numpy()
            edge_attr = graph_data.edge_attr.cpu().numpy()
            edge_labels = graph_data.edge_labels.cpu().numpy()
            
            if selected_edge_idx < len(edge_labels):
                source = edge_index[0, selected_edge_idx]
                target = edge_index[1, selected_edge_idx]
                is_fraud = edge_labels[selected_edge_idx]
                features = edge_attr[selected_edge_idx]
                
                st.markdown(f"**Nó de Origem:** {source}")
                st.markdown(f"**Nó de Destino:** {target}")
                st.markdown(f"**É Fraude:** {'Sim' if is_fraud else 'Não'}")
                
                # st.markdown("**Edge Features:**")
                # for i, feature_name in enumerate(explorer.edge_features):
                #     if i < len(features):
                #         st.markdown(f"- {feature_name}: {features[i]:.4f}")
            else:
                st.error(f"Índice da aresta {selected_edge_idx} está fora do intervalo.")
            
            # Instructions
            st.markdown("### 📖 Instruções")
            st.markdown("""
            1. **Selecione uma aresta** digitando seu índice na barra lateral
            2. **Clique em 'Obter Aresta'** para visualizar a vizinhança de 2 saltos
            3. **Explore o grafo** passando o mouse sobre nós e arestas
            4. **Clique em 'Executar Inferência'** para rodar o modelo GNN
            5. **Veja os resultados** na seção de inferência
            """)
            
            # Graph legend
            st.markdown("### 🎨 Legenda do Grafo")
            st.markdown("""
            - **Linhas amarelas**: Aresta selecionada
            - **Linhas vermelhas**: Transações fraudulentas
            - **Linhas cinzas**: Transações normais
            - **Nós roxos**: Clientes
            - **Nós amarelos**: Comerciantes
            """)
        else:
            st.info("Selecione uma aresta para ver informações aqui.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Explorador de Inferência GNN - Construído com Streamlit e PyTorch Geometric
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 