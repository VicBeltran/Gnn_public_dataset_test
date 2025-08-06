import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.transforms import RandomNodeSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score, f1_score
import warnings
import optuna
import json
from datetime import datetime
from training_pipeline import GenericTrainingPipeline, FocalLoss
from models import ImprovedEdgeGraphSAGE
import networkx as nx
import random
import torch.nn as nn
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

class GNNTrainingWrapper:
    """Generic wrapper for GNN training that handles graph construction from tabular data"""
    
    def __init__(self, random_state=42, max_subgraph_size=200, num_hops=1):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.node_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
        self.max_subgraph_size = max_subgraph_size
        self.num_hops = num_hops
        
    def sample_subgraph_around_edges(self, data, target_edge_indices, max_nodes=None):
        """
        Sample a small subgraph around specific edges using PyG's k_hop_subgraph
        
        Args:
            data: Full graph data
            target_edge_indices: Indices of edges to sample around
            max_nodes: Maximum number of nodes in subgraph (default: self.max_subgraph_size)
        """
        if max_nodes is None:
            max_nodes = self.max_subgraph_size
            
        # Get the nodes involved in target edges
        target_nodes = set()
        for edge_idx in target_edge_indices:
            source, target = data.edge_index[:, edge_idx].cpu().numpy()
            target_nodes.add(int(source))
            target_nodes.add(int(target))
        
        # Convert to list for k_hop_subgraph
        node_idx = list(target_nodes)
        
        # Use PyG's k_hop_subgraph with relabel_nodes=True
        from torch_geometric.utils import k_hop_subgraph
        
        node_mask, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
            node_idx=node_idx,
            num_hops=self.num_hops,
            edge_index=data.edge_index,
            relabel_nodes=True,  # This renumbers nodes to [0..N_sub-1]
            num_nodes=data.num_nodes
        )
        
        # Filter edge attributes and labels
        edge_attr_sub = data.edge_attr[edge_mask]
        edge_labels_sub = data.edge_labels[edge_mask]
        
        # Create subgraph data
        subgraph_data = Data(
            x=data.x[node_mask],  # Filter node features
            edge_index=edge_index_sub,
            edge_attr=edge_attr_sub,
            edge_labels=edge_labels_sub
        )
        
        # Store the node mask for heterogeneous models
        subgraph_data.node_mask = node_mask
        subgraph_data.original_num_nodes = data.num_nodes
        
        return subgraph_data
    
    def split_graph_data_with_sampling(self, data, test_size=0.2, val_size=0.2, random_state=42):
        """Split graph data into train/val/test sets with subgraph sampling"""
        print("Splitting graph data with subgraph sampling...")
        
        # Get edge indices
        num_edges = data.edge_index.shape[1]
        indices = np.arange(num_edges)
        
        # Split indices
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state, 
            stratify=data.edge_labels.cpu().numpy()
        )
        
        train_idx, val_idx = train_test_split(
            train_idx, test_size=val_size/(1-test_size), random_state=random_state,
            stratify=data.edge_labels.cpu().numpy()[train_idx]
        )
        
        # Sample subgraphs for each split with smaller sizes
        print(f"Sampling training subgraph (max {self.max_subgraph_size} nodes)...")
        train_data = self.sample_subgraph_around_edges(data, train_idx, max_nodes=self.max_subgraph_size)
        if train_data is None:
            # Fallback to full graph if sampling fails
            train_data = Data(
                x=data.x,
                edge_index=data.edge_index[:, train_idx],
                edge_attr=data.edge_attr[train_idx],
                edge_labels=data.edge_labels[train_idx]
            )
        
        print(f"Sampling validation subgraph (max {self.max_subgraph_size} nodes)...")
        val_data = self.sample_subgraph_around_edges(data, val_idx, max_nodes=self.max_subgraph_size)
        if val_data is None:
            val_data = Data(
                x=data.x,
                edge_index=data.edge_index[:, val_idx],
                edge_attr=data.edge_attr[val_idx],
                edge_labels=data.edge_labels[val_idx]
            )
        
        print(f"Sampling test subgraph (max {self.max_subgraph_size} nodes)...")
        test_data = self.sample_subgraph_around_edges(data, test_idx, max_nodes=self.max_subgraph_size)
        if test_data is None:
            test_data = Data(
                x=data.x,
                edge_index=data.edge_index[:, test_idx],
                edge_attr=data.edge_attr[test_idx],
                edge_labels=data.edge_labels[test_idx]
            )
        
        # Sort for LSTM compatibility
        train_data = train_data.sort(sort_by_row=False)
        val_data = val_data.sort(sort_by_row=False)
        test_data = test_data.sort(sort_by_row=False)
        
        print(f"Train subgraph: {train_data.num_nodes} nodes, {train_data.num_edges} edges")
        print(f"Validation subgraph: {val_data.num_nodes} nodes, {val_data.num_edges} edges")
        print(f"Test subgraph: {test_data.num_nodes} nodes, {test_data.num_edges} edges")
        
        return train_data, val_data, test_data
    
    def build_graph_from_tabular(self, df, node_features, edge_features, target_col, 
                                source_node_col=None, target_node_col=None, node_id_col=None,
                                source_node_features=None, target_node_features=None):
        """
        Build graph from tabular data with generic node approach
        
        Args:
            df: DataFrame with transaction/interaction data
            node_features: List of common node feature columns (deprecated, use source_node_features and target_node_features)
            edge_features: List of edge feature columns  
            target_col: Target column name
            source_node_col: Source node ID column (if None, will create synthetic IDs)
            target_node_col: Target node ID column (if None, will create synthetic IDs)
            node_id_col: Single node ID column (alternative to source/target approach)
            source_node_features: List of features specific to source nodes
            target_node_features: List of features specific to target nodes
        """
        print("Building graph from tabular data...")
        
        # Handle different node ID approaches
        if node_id_col is not None:
            # Single node ID approach - create edges between different nodes
            print("Using single node ID approach")
            df = df.copy()
            
            # Create synthetic target nodes if not provided
            if target_node_col is None:
                df['target_node'] = df[node_id_col].apply(lambda x: f"target_{x}")
                target_node_col = 'target_node'
            
            source_node_col = node_id_col
        else:
            # Dual node approach (source -> target)
            if source_node_col is None:
                df['source_node'] = range(len(df))
                source_node_col = 'source_node'
            
            if target_node_col is None:
                df['target_node'] = range(len(df))
                target_node_col = 'target_node'
        
        # Handle node features - use specific features if provided, otherwise fall back to common features
        if source_node_features is None:
            source_node_features = node_features
        if target_node_features is None:
            target_node_features = node_features
            
        print(f"Source node features: {source_node_features}")
        print(f"Target node features: {target_node_features}")
        
        # Get unique nodes
        source_nodes = df[source_node_col].unique()
        target_nodes = df[target_node_col].unique()
        
        print(f"Unique source nodes: {len(source_nodes)}")
        print(f"Unique target nodes: {len(target_nodes)}")
        print(f"Total interactions: {len(df)}")
        
        # Create mappings
        source_map = {nid: i for i, nid in enumerate(source_nodes)}
        target_map = {nid: i+len(source_nodes) for i, nid in enumerate(target_nodes)}
        
        # Build edge index and attributes
        edge_index = [[], []]
        edge_attr = []
        edge_labels = []
        
        for idx, row in df.iterrows():
            s_idx = source_map[row[source_node_col]]
            t_idx = target_map[row[target_node_col]]
            
            # Add edge from source to target
            edge_index[0].append(s_idx)
            edge_index[1].append(t_idx)
            
            # Add edge features
            edge_feat = [row[col] for col in edge_features]
            edge_attr.append(edge_feat)
            
            # Add target
            edge_labels.append(row[target_col])
        
        # Convert to tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        edge_labels = torch.tensor(edge_labels, dtype=torch.long)
        
        # Build node features with different features for source and target nodes
        node_features = self._build_node_features_heterogeneous(
            df, source_nodes, target_nodes, 
            source_node_features, target_node_features,
            source_node_col, target_node_col
        )
        
        # Normalize features
        edge_attr = torch.tensor(self.edge_scaler.fit_transform(edge_attr), dtype=torch.float)
        node_features = torch.tensor(self.node_scaler.fit_transform(node_features), dtype=torch.float)
        
        # Create Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_labels=edge_labels
        )
        
        # Sort for LSTM compatibility
        data = data.sort(sort_by_row=False)
        
        print(f"Graph created:")
        print(f"  Nodes: {data.num_nodes}")
        print(f"  Edges: {data.num_edges}")
        print(f"  Node features: {data.num_node_features}")
        print(f"  Edge features: {data.edge_attr.shape[1]}")
        print(f"  Fraud rate: {data.edge_labels.float().mean():.4f}")
        
        return data
    
    def inference_with_subgraph_sampling(self, model, full_data, target_edge_indices, batch_size=50):
        """
        Perform inference using subgraph sampling for production use
        
        Args:
            model: Trained GNN model
            full_data: Full graph data
            target_edge_indices: Indices of edges to predict
            batch_size: Number of edges to process in each batch
        """
        model.eval()
        predictions = []
        probabilities = []
        
        print(f"Performing inference on {len(target_edge_indices)} edges with subgraph sampling...")
        
        # Process edges in batches
        for i in range(0, len(target_edge_indices), batch_size):
            batch_indices = target_edge_indices[i:i+batch_size]
            
            # Sample small subgraph for this batch
            subgraph_data = self.sample_subgraph_around_edges(full_data, batch_indices, max_nodes=self.max_subgraph_size)
            
            if subgraph_data is None:
                # Fallback: use full graph
                subgraph_data = full_data
            
            # Move to device
            device = next(model.parameters()).device
            subgraph_data = subgraph_data.to(device)
            subgraph_data = subgraph_data.sort(sort_by_row=False)
            
            with torch.no_grad():
                # Check if we have heterogeneous node types
                use_heterogeneous = hasattr(self, 'source_feature_dim') and hasattr(self, 'target_feature_dim')
                
                if use_heterogeneous and self.source_feature_dim != self.target_feature_dim:
                    # Create heterogeneous data format using helper
                    x_dict, edge_index_dict, edge_attr_dict = self._create_heterogeneous_data(subgraph_data, device)
                    out = model(x_dict, edge_index_dict, edge_attr_dict)
                else:
                    out = model(subgraph_data.x, subgraph_data.edge_index, subgraph_data.edge_attr)
                
                batch_predictions = torch.softmax(out, dim=1)
                batch_probs = batch_predictions[:, 1].cpu().numpy()
                batch_preds = torch.argmax(batch_predictions, dim=1).cpu().numpy()
                
                # We need to map back to only the target edges
                # The subgraph contains all edges in the k-hop neighborhood, but we only want the original target edges
                # Find which edges in the subgraph correspond to our target edges
                target_edge_mask = self._get_target_edge_mask_in_subgraph(full_data, subgraph_data, batch_indices)
                
                if target_edge_mask is not None and np.any(target_edge_mask):
                    # Only keep predictions for the target edges
                    batch_probs = batch_probs[target_edge_mask]
                    batch_preds = batch_preds[target_edge_mask]
                    print(f"  Subgraph had {len(batch_probs)} target edges out of {len(target_edge_mask)} total edges")
                else:
                    print(f"  Warning: Could not map target edges or no target edges found, using all {len(batch_probs)} predictions")
                    # If we can't map target edges, we need to ensure we have the right number of predictions
                    # This is a fallback - ideally we should have the correct mapping
                    if len(batch_probs) != len(batch_indices):
                        print(f"  Warning: Prediction count mismatch. Expected {len(batch_indices)}, got {len(batch_probs)}")
                        # Take only the first len(batch_indices) predictions as fallback
                        batch_probs = batch_probs[:len(batch_indices)]
                        batch_preds = batch_preds[:len(batch_indices)]
                
                predictions.extend(batch_preds)
                probabilities.extend(batch_probs)
        
        # Ensure we have the correct number of predictions
        if len(predictions) != len(target_edge_indices):
            print(f"Warning: Final prediction count mismatch. Expected {len(target_edge_indices)}, got {len(predictions)}")
            # Pad or truncate to match expected length
            if len(predictions) < len(target_edge_indices):
                # Pad with zeros if we have fewer predictions
                padding_length = len(target_edge_indices) - len(predictions)
                predictions.extend([0] * padding_length)
                probabilities.extend([0.0] * padding_length)
            else:
                # Truncate if we have more predictions
                predictions = predictions[:len(target_edge_indices)]
                probabilities = probabilities[:len(target_edge_indices)]
        
        return np.array(predictions), np.array(probabilities)
    
    def _get_target_edge_mask_in_subgraph(self, full_data, subgraph_data, target_edge_indices):
        """
        Find which edges in the subgraph correspond to the original target edges.
        This is needed because k_hop_subgraph includes all edges in the neighborhood,
        but we only want predictions for the original target edges.
        
        Args:
            full_data: Original full graph data
            subgraph_data: Subgraph data created by k_hop_subgraph
            target_edge_indices: Original target edge indices in the full graph
            
        Returns:
            Boolean mask indicating which edges in subgraph_data correspond to target edges
        """
        if subgraph_data is None or target_edge_indices is None:
            return None
            
        try:
            # Get the original edge indices that are in the subgraph
            # The subgraph_data.edge_index contains the relabeled node indices
            # We need to map back to the original edge indices
            
            target_edge_mask = []
            
            # Create a mapping from subgraph node indices to original node indices
            if hasattr(subgraph_data, 'node_mask') and subgraph_data.node_mask is not None:
                # Use the explicit mapping returned by k_hop_subgraph
                if hasattr(subgraph_data, 'mapping'):
                    # Use the mapping tensor directly (more reliable)
                    node_mapping = subgraph_data.mapping
                else:
                    # Fallback to using node_mask
                    node_mapping = torch.where(subgraph_data.node_mask)[0]
            else:
                # If no node_mask, assume direct mapping (fallback)
                node_mapping = None
            
            # For each edge in the subgraph, check if it corresponds to a target edge
            for i in range(subgraph_data.edge_index.shape[1]):
                subgraph_source, subgraph_target = subgraph_data.edge_index[:, i].cpu().numpy()
                
                # Map back to original node indices
                if node_mapping is not None:
                    try:
                        # Ensure indices are within bounds
                        if (subgraph_source < len(node_mapping) and subgraph_target < len(node_mapping)):
                            original_source = node_mapping[subgraph_source].item()
                            original_target = node_mapping[subgraph_target].item()
                        else:
                            # Skip this edge if indices are out of bounds
                            target_edge_mask.append(False)
                            continue
                    except (IndexError, RuntimeError) as e:
                        # If mapping fails, skip this edge
                        print(f"Warning: Failed to map edge {i} in subgraph: {e}")
                        target_edge_mask.append(False)
                        continue
                else:
                    # If no node_mask, assume direct mapping (fallback)
                    original_source = subgraph_source
                    original_target = subgraph_target
                
                # Check if this edge exists in the original target edges
                is_target_edge = False
                for target_idx in target_edge_indices:
                    orig_source, orig_target = full_data.edge_index[:, target_idx].cpu().numpy()
                    if (original_source == orig_source and original_target == orig_target) or \
                       (original_source == orig_target and original_target == orig_source):
                        is_target_edge = True
                        break
                
                target_edge_mask.append(is_target_edge)
            
            return np.array(target_edge_mask)
            
        except Exception as e:
            print(f"Error in _get_target_edge_mask_in_subgraph: {e}")
            print("Falling back to using all subgraph edges...")
            # Return None to indicate we should use all edges
            return None
    
    def demonstrate_production_inference(self, model, full_data, num_test_edges=100):
        """
        Demonstrate production inference with subgraph sampling
        
        Args:
            model: Trained GNN model
            full_data: Full graph data
            num_test_edges: Number of edges to test inference on
        """
        print("\n" + "="*50)
        print("PRODUCTION INFERENCE DEMONSTRATION")
        print("="*50)
        
        # Select random edges for inference
        num_edges = full_data.edge_index.shape[1]
        test_edge_indices = np.random.choice(num_edges, min(num_test_edges, num_edges), replace=False)
        
        print(f"Testing inference on {len(test_edge_indices)} edges...")
        
        # Perform inference with subgraph sampling
        start_time = datetime.now()
        try:
            predictions, probabilities = self.inference_with_subgraph_sampling(
                model, full_data, test_edge_indices, batch_size=50
            )
            end_time = datetime.now()
            
            inference_time = (end_time - start_time).total_seconds()
            avg_time_per_edge = inference_time / len(test_edge_indices)
            
            print(f"Inference completed in {inference_time:.2f} seconds")
            print(f"Average time per edge: {avg_time_per_edge:.4f} seconds")
            print(f"Throughput: {len(test_edge_indices)/inference_time:.2f} edges/second")
            
            # Calculate metrics
            true_labels = full_data.edge_labels[test_edge_indices].cpu().numpy()
            
            # Ensure we have the same number of predictions and true labels
            if len(predictions) != len(true_labels):
                print(f"Warning: Prediction count mismatch. Expected {len(true_labels)}, got {len(predictions)}")
                min_len = min(len(predictions), len(true_labels))
                predictions = predictions[:min_len]
                probabilities = probabilities[:min_len]
                true_labels = true_labels[:min_len]
            
            accuracy = np.mean(true_labels == predictions)
            auc = roc_auc_score(true_labels, probabilities)
            
        except Exception as e:
            print(f"Error during inference: {e}")
            print("Falling back to simple evaluation...")
            
            # Fallback: use simple evaluation without subgraph sampling
            model.eval()
            device = next(model.parameters()).device
            full_data = full_data.to(device)
            
            with torch.no_grad():
                if hasattr(self, 'source_feature_dim') and hasattr(self, 'target_feature_dim'):
                    x_dict, edge_index_dict, edge_attr_dict = self._create_heterogeneous_data(full_data, device)
                    out = model(x_dict, edge_index_dict, edge_attr_dict)
                else:
                    out = model(full_data.x, full_data.edge_index, full_data.edge_attr)
                
                predictions = torch.softmax(out, dim=1)
                probabilities = predictions[:, 1].cpu().numpy()
                predictions = torch.argmax(predictions, dim=1).cpu().numpy()
                
                # Get predictions for test edges only
                predictions = predictions[test_edge_indices]
                probabilities = probabilities[test_edge_indices]
                true_labels = full_data.edge_labels[test_edge_indices].cpu().numpy()
                
                accuracy = np.mean(true_labels == predictions)
                auc = roc_auc_score(true_labels, probabilities)
                inference_time = 0.0  # We don't have timing for fallback
        
        print(f"Production Inference Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  ROC-AUC: {auc:.4f}")
        print(f"  Fraud Rate: {true_labels.mean():.4f}")
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'true_labels': true_labels,
            'inference_time': inference_time,
            'throughput': len(test_edge_indices)/inference_time,
            'accuracy': accuracy,
            'auc': auc
        }
    
    def _build_node_features(self, df, source_nodes, target_nodes, node_features, source_node_col, target_node_col):
        """Build node features for source and target nodes (legacy method)"""
        print("Building node features...")
        
        # Source node features
        source_features = []
        for node in source_nodes:
            node_data = df[df[source_node_col] == node]
            
            # Aggregate features
            feat_vector = []
            for feat in node_features:
                if feat in node_data.columns:
                    feat_vector.append(node_data[feat].mean())
                else:
                    feat_vector.append(0.0)  # Default value
            
            # Add interaction count and fraud rate
            feat_vector.append(len(node_data))
            feat_vector.append(node_data['is_fraud'].mean() if 'is_fraud' in node_data.columns else 0.0)
            
            source_features.append(feat_vector)
        
        # Target node features  
        target_features = []
        for node in target_nodes:
            node_data = df[df[target_node_col] == node]
            
            # Aggregate features
            feat_vector = []
            for feat in node_features:
                if feat in node_data.columns:
                    feat_vector.append(node_data[feat].mean())
                else:
                    feat_vector.append(0.0)  # Default value
            
            # Add interaction count and fraud rate
            feat_vector.append(len(node_data))
            feat_vector.append(node_data['is_fraud'].mean() if 'is_fraud' in node_data.columns else 0.0)
            
            target_features.append(feat_vector)
        
        # Combine source and target features
        all_node_features = source_features + target_features
        return np.array(all_node_features)
    
    def _build_node_features_heterogeneous(self, df, source_nodes, target_nodes, 
                                         source_node_features, target_node_features,
                                         source_node_col, target_node_col):
        """
        Build heterogeneous node features without padding - each node type keeps its original features.
        This approach works with the new heterogeneous model that handles different feature dimensions.
        """
        print("Building heterogeneous node features...")
        
        # Source node features (keep original dimensions)
        source_features = []
        for node in source_nodes:
            node_data = df[df[source_node_col] == node]
            
            feat_vector = []
            for feat in source_node_features:
                if feat in node_data.columns:
                    feat_vector.append(node_data[feat].mean())
                else:
                    feat_vector.append(0.0)
            
            # Add interaction count and fraud rate
            feat_vector.append(len(node_data))
            feat_vector.append(node_data['is_fraud'].mean() if 'is_fraud' in node_data.columns else 0.0)
            
            source_features.append(feat_vector)
        
        # Target node features (keep original dimensions)
        target_features = []
        for node in target_nodes:
            node_data = df[df[target_node_col] == node]
            
            feat_vector = []
            for feat in target_node_features:
                if feat in node_data.columns:
                    feat_vector.append(node_data[feat].mean())
                else:
                    feat_vector.append(0.0)
            
            # Add interaction count and fraud rate
            feat_vector.append(len(node_data))
            feat_vector.append(node_data['is_fraud'].mean() if 'is_fraud' in node_data.columns else 0.0)
            
            target_features.append(feat_vector)
        
        # Convert to numpy arrays
        source_features = np.array(source_features)
        target_features = np.array(target_features)
        
        print(f"Source node features: {source_features.shape}")
        print(f"Target node features: {target_features.shape}")
        
        # Store original feature dimensions for the heterogeneous model
        self.source_feature_dim = source_features.shape[1]
        self.target_feature_dim = target_features.shape[1]
        
        # Store the separate feature arrays for the new heterogeneous approach
        self.source_features = source_features
        self.target_features = target_features
        
        # For the new heterogeneous model, we need to create separate feature dictionaries
        # but for backward compatibility with the current graph building, we'll also create the unified tensor
        # Pad to the same dimension for the unified tensor (for backward compatibility)
        max_dim = max(source_features.shape[1], target_features.shape[1])
        
        if source_features.shape[1] < max_dim:
            padding = np.zeros((source_features.shape[0], max_dim - source_features.shape[1]))
            source_features = np.hstack([source_features, padding])
        
        if target_features.shape[1] < max_dim:
            padding = np.zeros((target_features.shape[0], max_dim - target_features.shape[1]))
            target_features = np.hstack([target_features, padding])
        
        # Combine features for unified tensor
        all_node_features = np.vstack([source_features, target_features])
        
        print(f"Unified node features: {all_node_features.shape}")
        
        return all_node_features
    
    def train_gnn_with_optimization(self, train_data, val_data, test_data, 
                                   sampling_method=None, n_trials=50, final_training=True):
        """Train GNN with hyperparameter optimization using subgraph sampling"""
        print("Training GNN with optimization and subgraph sampling...")
        
        # Optimize hyperparameters
        if n_trials > 0:
            print("Optimizing hyperparameters...")
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=self.random_state)
            )
            
            def objective(trial):
                return self._objective_gnn(trial, train_data, val_data)
            
            study.optimize(objective, n_trials=n_trials)
            best_params = study.best_params
            print(f"Best parameters: {best_params}")
            print(f"Best validation AUC: {study.best_value:.4f}")
        else:
            best_params = None
        
        # Final training
        if final_training:
            print("Training final model...")
            model, train_losses, val_losses = self._train_gnn_model(
                train_data, val_data, best_params
            )
            
            # Evaluate
            results = self._evaluate_gnn_model(model, test_data)
            
            return model, results, best_params
        
        return None, None, best_params
    
    def _objective_gnn(self, trial, data_train, data_val):
        """Optuna objective function for GNN hyperparameter optimization"""
        params = {
            'hidden_channels': trial.suggest_int('hidden_channels', 64, 256),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'lr': trial.suggest_float('lr', 1e-4, 5e-3, log=True),  # Narrowed range for better convergence
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),  # Reduced max
            'alpha': trial.suggest_float('alpha', 0.5, 2.0),
            'gamma': trial.suggest_float('gamma', 1.0, 4.0),
        }
        
        # Get device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Apply SMOTE with fixed ratio for slight imbalance
        data_train_resampled = self.apply_smote_to_edges(data_train, sampling_ratio=0.7)
        
        # Move data to device and ensure proper sorting
        data_train_resampled = data_train_resampled.to(device)
        data_val = data_val.to(device)
        
        # Always sort edge_index for LSTM aggregation
        data_train_resampled = data_train_resampled.sort(sort_by_row=False)
        data_val = data_val.sort(sort_by_row=False)
        
        # Check if we have heterogeneous node types
        use_heterogeneous = hasattr(self, 'source_feature_dim') and hasattr(self, 'target_feature_dim')
        
        if use_heterogeneous and self.source_feature_dim != self.target_feature_dim:
            # Use heterogeneous model
            from models import HeterogeneousEdgeGraphSAGE
            model = HeterogeneousEdgeGraphSAGE(
                source_feature_dim=self.source_feature_dim,
                target_feature_dim=self.target_feature_dim,
                hidden_channels=params['hidden_channels'],
                out_channels=2,  # Binary classification
                edge_dim=data_train_resampled.edge_attr.shape[1],
                dropout=params['dropout']
            ).to(device)
        else:
            # Use standard model
            from models import ImprovedEdgeGraphSAGE
            model = ImprovedEdgeGraphSAGE(
                in_channels=data_train_resampled.num_node_features,
                hidden_channels=params['hidden_channels'],
                out_channels=2,  # Binary classification
                edge_dim=data_train_resampled.edge_attr.shape[1],
                dropout=params['dropout']
            ).to(device)
        
        # Loss function with focal loss - use standard cross entropy for better stability
        # Calculate class weights based on data distribution
        labels = data_train_resampled.edge_labels.cpu().numpy()
        num_pos = np.sum(labels == 1)
        num_neg = np.sum(labels == 0)
        pos_weight_val = num_neg / num_pos if num_pos > 0 else 1.0
        
        # Use standard cross entropy with class weights for better stability
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, pos_weight_val], dtype=torch.float32).to(device)
        ).to(device)
        
        # Optimizer with better parameters
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['lr'],
            weight_decay=params['weight_decay'],
            betas=(0.9, 0.999),  # Explicit betas for better convergence
            eps=1e-8
        )
        
        # Training loop with improved parameters
        best_val_auc = 0.0  # Track best AUC instead of loss
        best_model_state = None
        patience = 25  # Increased patience for final training
        patience_counter = 0
        training_losses = []
        validation_aucs = []
        
        # Learning rate scheduler with better parameters
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=12, factor=0.7, verbose=True, min_lr=1e-6
        )
        
        for epoch in range(100):  # Increased epochs for optimization
            model.train()
            optimizer.zero_grad()
            
            if use_heterogeneous and self.source_feature_dim != self.target_feature_dim:
                # Create heterogeneous data format using helper
                x_dict, edge_index_dict, edge_attr_dict = self._create_heterogeneous_data(data_train_resampled, device)
                out = model(x_dict, edge_index_dict, edge_attr_dict)
            else:
                out = model(data_train_resampled.x, data_train_resampled.edge_index, data_train_resampled.edge_attr)
            
            loss = criterion(out, data_train_resampled.edge_labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                if use_heterogeneous and self.source_feature_dim != self.target_feature_dim:
                    # Create heterogeneous data format using helper
                    x_dict, edge_index_dict, edge_attr_dict = self._create_heterogeneous_data(data_val, device)
                    val_out = model(x_dict, edge_index_dict, edge_attr_dict)
                else:
                    val_out = model(data_val.x, data_val.edge_index, data_val.edge_attr)
                
                val_predictions = torch.softmax(val_out, dim=1)
                val_probs = val_predictions[:, 1].cpu().numpy()
                val_auc = roc_auc_score(data_val.edge_labels.cpu().numpy(), val_probs)
                
                # Learning rate scheduling based on AUC
                scheduler.step(val_auc)
                
                # Early stopping based on validation AUC
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Print progress every 20 epochs
                if epoch % 20 == 0:
                    print(f'Trial epoch {epoch+1}, Train Loss: {loss.item():.6f}, Val AUC: {val_auc:.4f}')
                
                if patience_counter >= patience:
                    break
        
        return best_val_auc  # Return AUC directly for maximization
    
    def _train_gnn_model(self, data_train, data_val, params=None):
        """Train GNN model with optimal parameters"""
        if params is None:
            params = {
                'hidden_channels': 128,
                'dropout': 0.2,
                'lr': 0.001,
                'weight_decay': 0.0001,
                'alpha': 1.0,
                'gamma': 2.0,
            }
        
        print("Training GNN model with parameters:", params)
        
        # Get device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Apply SMOTE with fixed ratio for slight imbalance
        data_train_resampled = self.apply_smote_to_edges(data_train, sampling_ratio=0.7)
        
        # Move data to device and ensure proper sorting
        data_train_resampled = data_train_resampled.to(device)
        data_val = data_val.to(device)
        
        # Always sort edge_index for LSTM aggregation
        data_train_resampled = data_train_resampled.sort(sort_by_row=False)
        data_val = data_val.sort(sort_by_row=False)
        
        # Check if we have heterogeneous node types
        use_heterogeneous = hasattr(self, 'source_feature_dim') and hasattr(self, 'target_feature_dim')
        
        if use_heterogeneous and self.source_feature_dim != self.target_feature_dim:
            # Use heterogeneous model
            from models import HeterogeneousEdgeGraphSAGE
            model = HeterogeneousEdgeGraphSAGE(
                source_feature_dim=self.source_feature_dim,
                target_feature_dim=self.target_feature_dim,
                hidden_channels=params['hidden_channels'],
                out_channels=2,
                edge_dim=data_train_resampled.edge_attr.shape[1],
                dropout=params['dropout']
            ).to(device)
        else:
            # Use standard model
            from models import ImprovedEdgeGraphSAGE
            model = ImprovedEdgeGraphSAGE(
                in_channels=data_train_resampled.num_node_features,
                hidden_channels=params['hidden_channels'],
                out_channels=2,
                edge_dim=data_train_resampled.edge_attr.shape[1],
                dropout=params['dropout']
            ).to(device)
        
        # Loss function with standard cross entropy for consistency with optimization
        # Calculate class weights based on data distribution
        labels = data_train_resampled.edge_labels.cpu().numpy()
        num_pos = np.sum(labels == 1)
        num_neg = np.sum(labels == 0)
        pos_weight_val = num_neg / num_pos if num_pos > 0 else 1.0
        
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, pos_weight_val], dtype=torch.float32).to(device)
        ).to(device)
        
        # Optimizer with better parameters
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['lr'],
            weight_decay=params['weight_decay'],
            betas=(0.9, 0.999),  # Explicit betas for better convergence
            eps=1e-8
        )
        
        # Training loop with improved parameters
        best_val_auc = 0.0  # Track best AUC instead of loss
        best_model_state = None
        patience = 25  # Increased patience for final training
        patience_counter = 0
        training_losses = []
        validation_aucs = []
        
        # Learning rate scheduler with better parameters
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=12, factor=0.7, verbose=True, min_lr=1e-6
        )
        
        for epoch in range(450):  # More epochs for final training
            model.train()
            optimizer.zero_grad()
            
            if use_heterogeneous and self.source_feature_dim != self.target_feature_dim:
                # Create heterogeneous data format using helper
                x_dict, edge_index_dict, edge_attr_dict = self._create_heterogeneous_data(data_train_resampled, device)
                out = model(x_dict, edge_index_dict, edge_attr_dict)
            else:
                out = model(data_train_resampled.x, data_train_resampled.edge_index, data_train_resampled.edge_attr)
            
            loss = criterion(out, data_train_resampled.edge_labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                if use_heterogeneous and self.source_feature_dim != self.target_feature_dim:
                    # Create heterogeneous data format using helper
                    x_dict, edge_index_dict, edge_attr_dict = self._create_heterogeneous_data(data_val, device)
                    val_out = model(x_dict, edge_index_dict, edge_attr_dict)
                else:
                    val_out = model(data_val.x, data_val.edge_index, data_val.edge_attr)
                
                val_predictions = torch.softmax(val_out, dim=1)
                val_probs = val_predictions[:, 1].cpu().numpy()
                val_auc = roc_auc_score(data_val.edge_labels.cpu().numpy(), val_probs)
                
                # Learning rate scheduling based on AUC
                scheduler.step(val_auc)
                
                # Track losses and AUCs
                training_losses.append(loss.item())
                validation_aucs.append(val_auc)
                
                # Early stopping and best model tracking based on AUC
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model_state = model.state_dict().copy()  # Save best model state
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if epoch % 25 == 0:  # Print every 25 epochs
                    print(f'Epoch {epoch+1}, Train Loss: {loss.item():.6f}, Val AUC: {val_auc:.4f}')
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model and save it
        model.load_state_dict(best_model_state)
        
        # Save only the best model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f'./models/gnn_models/gnn_model_{timestamp}.pt'
        torch.save(model.state_dict(), model_path)
        
        # Save parameters
        params_path = f'./models/gnn_models/gnn_params_{timestamp}.json'
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=4)
        
        print(f"Best GNN model saved to {model_path}")
        print(f"Best validation AUC: {best_val_auc:.6f}")
        
        return model, training_losses, validation_aucs
    
    def _evaluate_gnn_model(self, model, test_data):
        """Evaluate GNN model performance"""
        # Get device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Move test data to device and ensure proper sorting
        test_data = test_data.to(device)
        test_data = test_data.sort(sort_by_row=False)
        
        # Check if we have heterogeneous node types
        use_heterogeneous = hasattr(self, 'source_feature_dim') and hasattr(self, 'target_feature_dim')
        
        model.eval()
        with torch.no_grad():
            if use_heterogeneous and self.source_feature_dim != self.target_feature_dim:
                # Create heterogeneous data format using helper
                x_dict, edge_index_dict, edge_attr_dict = self._create_heterogeneous_data(test_data, device)
                out = model(x_dict, edge_index_dict, edge_attr_dict)
            else:
                out = model(test_data.x, test_data.edge_index, test_data.edge_attr)
            
            predictions = torch.softmax(out, dim=1)
            y_pred_proba = predictions[:, 1].cpu().numpy()
            y_pred = torch.argmax(predictions, dim=1).cpu().numpy()
            y_test = test_data.edge_labels.cpu().numpy()
        
        # Calculate metrics
        accuracy = (y_test == y_pred).mean()
        auc = roc_auc_score(y_test, y_pred_proba)
        ap = average_precision_score(y_test, y_pred_proba)
        
        # Find optimal threshold
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        # Apply optimal threshold
        y_pred_optimal = (y_pred_proba > optimal_threshold).astype(int)
        
        # Calculate additional metrics
        f1 = f1_score(y_test, y_pred_optimal)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_optimal)
        
        results = {
            'accuracy': accuracy,
            'auc': auc,
            'ap': ap,
            'f1_score': f1,
            'optimal_threshold': optimal_threshold,
            'confusion_matrix': cm,
            'predictions': y_pred_optimal,
            'probabilities': y_pred_proba
        }
        
        # Print results
        print(f"\nGNN Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC-AUC: {auc:.4f}")
        print(f"Average Precision: {ap:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Optimal Threshold: {optimal_threshold:.3f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred_optimal, target_names=['Not Fraud', 'Fraud']))
        
        print(f"\nConfusion Matrix:")
        print(f"True Negatives: {cm[0,0]}")
        print(f"False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}")
        print(f"True Positives: {cm[1,1]}")
        
        return results
    
    def save_results(self, results, filename=None):
        """Save evaluation results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'./outputs/gnn_results_{timestamp}.json'
        
        # Convert numpy arrays and other non-serializable types to JSON serializable
        results_serializable = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                results_serializable[key] = value.tolist()
            elif isinstance(value, np.integer):
                results_serializable[key] = int(value)
            elif isinstance(value, np.floating):
                results_serializable[key] = float(value)
            elif isinstance(value, np.bool_):
                results_serializable[key] = bool(value)
            else:
                results_serializable[key] = value
        
        with open(filename, 'w') as f:
            json.dump(results_serializable, f, indent=4)
        
        print(f"Results saved to {filename}")

    def _create_heterogeneous_data(self, data, device):
        """
        Create heterogeneous data format for the model with proper lock-step subgraph handling
        
        Args:
            data: PyG Data object (could be subgraph or full graph)
            device: Target device
            
        Returns:
            x_dict, edge_index_dict, edge_attr_dict for heterogeneous model
        """
        if not hasattr(self, 'source_features') or not hasattr(self, 'target_features'):
            raise ValueError("Heterogeneous features not initialized")
        
        # Check if we're using a subgraph
        total_original_nodes = len(self.source_features) + len(self.target_features)
        
        if hasattr(data, 'node_mask') and data.num_nodes < total_original_nodes:
            # We're using a subgraph - need to map node indices properly
            # The node_mask tells us which original nodes are in the subgraph
            node_mask = data.node_mask
            num_source_original = len(self.source_features)
            
            # Split the node mask into source and target nodes
            source_mask = node_mask < num_source_original
            target_mask = node_mask >= num_source_original
            
            # Get the source and target node indices from the original graph
            source_nodes_original = node_mask[source_mask]
            target_nodes_original = node_mask[target_mask] - num_source_original  # Adjust for target indexing
            
            # Check if we have both source and target nodes
            if len(source_nodes_original) == 0 or len(target_nodes_original) == 0:
                # Fallback to full graph if subgraph is incomplete
                x_dict = {
                    'source': torch.tensor(self.source_features, dtype=torch.float).to(device),
                    'target': torch.tensor(self.target_features, dtype=torch.float).to(device)
                }
            else:
                # Create feature tensors for the nodes in the subgraph
                source_features_subgraph = torch.tensor(
                    self.source_features[source_nodes_original], 
                    dtype=torch.float
                ).to(device)
                
                target_features_subgraph = torch.tensor(
                    self.target_features[target_nodes_original], 
                    dtype=torch.float
                ).to(device)
                
                x_dict = {
                    'source': source_features_subgraph,
                    'target': target_features_subgraph
                }
        else:
            # Using full graph
            x_dict = {
                'source': torch.tensor(self.source_features, dtype=torch.float).to(device),
                'target': torch.tensor(self.target_features, dtype=torch.float).to(device)
            }
        
        # Initialize edge dictionaries
        edge_index_dict = {}
        edge_attr_dict = {}
        
        # For subgraphs, we need to ensure edge indices are properly mapped to the subgraph node indices
        if hasattr(data, 'node_mask') and data.num_nodes < total_original_nodes:
            # The edge indices in the subgraph are already remapped to subgraph indices
            # But we need to ensure they reference the correct node types in the heterogeneous setup
            
            # Get the current edge indices
            edge_index = data.edge_index
            row, col = edge_index
            
            # Create mappings for source and target nodes in the subgraph
            node_mask = data.node_mask
            num_source_original = len(self.source_features)
            
            # Split the node mask into source and target nodes
            source_mask = node_mask < num_source_original
            target_mask = node_mask >= num_source_original
            
            # Get the subgraph indices for source and target nodes
            source_nodes_subgraph = torch.where(source_mask)[0]
            target_nodes_subgraph = torch.where(target_mask)[0]
            
            # Create mappings from subgraph indices to source/target indices
            source_subgraph_to_source = {subgraph_idx.item(): source_idx for source_idx, subgraph_idx in enumerate(source_nodes_subgraph)}
            target_subgraph_to_target = {subgraph_idx.item(): target_idx for target_idx, subgraph_idx in enumerate(target_nodes_subgraph)}
            
            # Initialize edge lists for each type
            source_to_source_edges = []
            target_to_target_edges = []
            source_to_target_edges = []
            target_to_source_edges = []
            
            # Remap edge indices to source/target indices within the heterogeneous setup
            for i in range(edge_index.size(1)):
                row_idx = row[i].item()
                col_idx = col[i].item()
                
                # Determine edge type based on node types
                row_is_source = row_idx in source_subgraph_to_source
                col_is_source = col_idx in source_subgraph_to_source
                row_is_target = row_idx in target_subgraph_to_target
                col_is_target = col_idx in target_subgraph_to_target
                
                if row_is_source and col_is_source:
                    # Source to source edge
                    source_to_source_edges.append((
                        source_subgraph_to_source[row_idx],
                        source_subgraph_to_source[col_idx],
                        i  # Original edge index for attributes
                    ))
                elif row_is_target and col_is_target:
                    # Target to target edge
                    target_to_target_edges.append((
                        target_subgraph_to_target[row_idx],
                        target_subgraph_to_target[col_idx],
                        i  # Original edge index for attributes
                    ))
                elif row_is_source and col_is_target:
                    # Source to target edge
                    source_to_target_edges.append((
                        source_subgraph_to_source[row_idx],
                        target_subgraph_to_target[col_idx],
                        i  # Original edge index for attributes
                    ))
                elif row_is_target and col_is_source:
                    # Target to source edge
                    target_to_source_edges.append((
                        target_subgraph_to_target[row_idx],
                        source_subgraph_to_source[col_idx],
                        i  # Original edge index for attributes
                    ))
            
            # Create edge index tensors for each type
            if source_to_source_edges:
                src_src_indices = torch.tensor([[e[0], e[1]] for e in source_to_source_edges], dtype=torch.long).t()
                edge_index_dict[('source', 'to', 'source')] = src_src_indices
                edge_attr_dict[('source', 'to', 'source')] = data.edge_attr[[e[2] for e in source_to_source_edges]]
            
            if target_to_target_edges:
                tgt_tgt_indices = torch.tensor([[e[0], e[1]] for e in target_to_target_edges], dtype=torch.long).t()
                edge_index_dict[('target', 'to', 'target')] = tgt_tgt_indices
                edge_attr_dict[('target', 'to', 'target')] = data.edge_attr[[e[2] for e in target_to_target_edges]]
            
            if source_to_target_edges:
                src_tgt_indices = torch.tensor([[e[0], e[1]] for e in source_to_target_edges], dtype=torch.long).t()
                edge_index_dict[('source', 'to', 'target')] = src_tgt_indices
                edge_attr_dict[('source', 'to', 'target')] = data.edge_attr[[e[2] for e in source_to_target_edges]]
            
            if target_to_source_edges:
                tgt_src_indices = torch.tensor([[e[0], e[1]] for e in target_to_source_edges], dtype=torch.long).t()
                edge_index_dict[('target', 'to', 'source')] = tgt_src_indices
                edge_attr_dict[('target', 'to', 'source')] = data.edge_attr[[e[2] for e in target_to_source_edges]]
            
            # Create empty tensors for missing edge types
            if ('source', 'to', 'source') not in edge_index_dict:
                edge_index_dict[('source', 'to', 'source')] = torch.empty((2, 0), dtype=torch.long)
                edge_attr_dict[('source', 'to', 'source')] = torch.empty((0, data.edge_attr.size(1)), dtype=torch.float)
            
            if ('target', 'to', 'target') not in edge_index_dict:
                edge_index_dict[('target', 'to', 'target')] = torch.empty((2, 0), dtype=torch.long)
                edge_attr_dict[('target', 'to', 'target')] = torch.empty((0, data.edge_attr.size(1)), dtype=torch.float)
            
            if ('source', 'to', 'target') not in edge_index_dict:
                edge_index_dict[('source', 'to', 'target')] = torch.empty((2, 0), dtype=torch.long)
                edge_attr_dict[('source', 'to', 'target')] = torch.empty((0, data.edge_attr.size(1)), dtype=torch.float)
            
            if ('target', 'to', 'source') not in edge_index_dict:
                edge_index_dict[('target', 'to', 'source')] = torch.empty((2, 0), dtype=torch.long)
                edge_attr_dict[('target', 'to', 'source')] = torch.empty((0, data.edge_attr.size(1)), dtype=torch.float)
        else:
            # Using full graph - create all edge types from the original edge_index
            edge_index = data.edge_index
            row, col = edge_index
            num_source_original = len(self.source_features)
            
            # Split edges by type
            source_to_source_edges = []
            target_to_target_edges = []
            source_to_target_edges = []
            target_to_source_edges = []
            
            for i in range(edge_index.size(1)):
                row_idx = row[i].item()
                col_idx = col[i].item()
                
                if row_idx < num_source_original and col_idx < num_source_original:
                    # Source to source edge
                    source_to_source_edges.append(i)
                elif row_idx >= num_source_original and col_idx >= num_source_original:
                    # Target to target edge
                    target_to_target_edges.append(i)
                elif row_idx < num_source_original and col_idx >= num_source_original:
                    # Source to target edge
                    source_to_target_edges.append(i)
                elif row_idx >= num_source_original and col_idx < num_source_original:
                    # Target to source edge
                    target_to_source_edges.append(i)
            
            # Create edge index tensors for each type
            if source_to_source_edges:
                src_src_indices = edge_index[:, source_to_source_edges]
                edge_index_dict[('source', 'to', 'source')] = src_src_indices
                edge_attr_dict[('source', 'to', 'source')] = data.edge_attr[source_to_source_edges]
            
            if target_to_target_edges:
                tgt_tgt_indices = edge_index[:, target_to_target_edges]
                # Adjust indices for target nodes
                tgt_tgt_indices[0] -= num_source_original
                tgt_tgt_indices[1] -= num_source_original
                edge_index_dict[('target', 'to', 'target')] = tgt_tgt_indices
                edge_attr_dict[('target', 'to', 'target')] = data.edge_attr[target_to_target_edges]
            
            if source_to_target_edges:
                src_tgt_indices = edge_index[:, source_to_target_edges]
                # Adjust target indices
                src_tgt_indices[1] -= num_source_original
                edge_index_dict[('source', 'to', 'target')] = src_tgt_indices
                edge_attr_dict[('source', 'to', 'target')] = data.edge_attr[source_to_target_edges]
            
            if target_to_source_edges:
                tgt_src_indices = edge_index[:, target_to_source_edges]
                # Adjust source indices
                tgt_src_indices[0] -= num_source_original
                edge_index_dict[('target', 'to', 'source')] = tgt_src_indices
                edge_attr_dict[('target', 'to', 'source')] = data.edge_attr[target_to_source_edges]
            
            # Create empty tensors for missing edge types
            if ('source', 'to', 'source') not in edge_index_dict:
                edge_index_dict[('source', 'to', 'source')] = torch.empty((2, 0), dtype=torch.long)
                edge_attr_dict[('source', 'to', 'source')] = torch.empty((0, data.edge_attr.size(1)), dtype=torch.float)
            
            if ('target', 'to', 'target') not in edge_index_dict:
                edge_index_dict[('target', 'to', 'target')] = torch.empty((2, 0), dtype=torch.long)
                edge_attr_dict[('target', 'to', 'target')] = torch.empty((0, data.edge_attr.size(1)), dtype=torch.float)
            
            if ('source', 'to', 'target') not in edge_index_dict:
                edge_index_dict[('source', 'to', 'target')] = torch.empty((2, 0), dtype=torch.long)
                edge_attr_dict[('source', 'to', 'target')] = torch.empty((0, data.edge_attr.size(1)), dtype=torch.float)
            
            if ('target', 'to', 'source') not in edge_index_dict:
                edge_index_dict[('target', 'to', 'source')] = torch.empty((2, 0), dtype=torch.long)
                edge_attr_dict[('target', 'to', 'source')] = torch.empty((0, data.edge_attr.size(1)), dtype=torch.float)
        
        return x_dict, edge_index_dict, edge_attr_dict

    def apply_smote_to_edges(self, data, sampling_ratio=1.0):
        """
        Apply SMOTE to edge data to handle imbalanced datasets
        
        Args:
            data: PyG Data object with edge features and labels
            sampling_ratio: Ratio of minority class to majority class (1.0 = balanced)
        
        Returns:
            Resampled PyG Data object
        """
        print("Applying SMOTE to edge data...")
        
        # Extract edge features and labels
        edge_features = data.edge_attr.cpu().numpy()
        edge_labels = data.edge_labels.cpu().numpy()
        
        print(f"Original edge distribution: {np.bincount(edge_labels)}")
        
        # Apply SMOTE
        smote = SMOTE(random_state=self.random_state, sampling_strategy=sampling_ratio)
        edge_features_resampled, edge_labels_resampled = smote.fit_resample(edge_features, edge_labels)
        
        print(f"Resampled edge distribution: {np.bincount(edge_labels_resampled)}")
        
        # Create new edge indices for synthetic edges
        original_num_edges = data.edge_index.shape[1]
        synthetic_num_edges = len(edge_labels_resampled) - original_num_edges
        
        if synthetic_num_edges > 0:
            # For heterogeneous graphs, we need to be careful about edge creation
            # We'll create synthetic edges that respect the source/target node structure
            num_source_nodes = len(self.source_features) if hasattr(self, 'source_features') else data.num_nodes // 2
            num_target_nodes = len(self.target_features) if hasattr(self, 'target_features') else data.num_nodes - num_source_nodes
            
            # Create synthetic edges that respect the heterogeneous structure
            synthetic_source_nodes = np.random.choice(num_source_nodes, synthetic_num_edges)
            synthetic_target_nodes = np.random.choice(num_target_nodes, synthetic_num_edges) + num_source_nodes
            
            # Create synthetic edge indices
            synthetic_edge_index = np.vstack([synthetic_source_nodes, synthetic_target_nodes])
            
            # Combine original and synthetic edges
            combined_edge_index = np.hstack([data.edge_index.cpu().numpy(), synthetic_edge_index])
            combined_edge_attr = torch.cat([
                data.edge_attr,
                torch.tensor(edge_features_resampled[original_num_edges:], dtype=torch.float32)
            ], dim=0)
            combined_edge_labels = torch.tensor(edge_labels_resampled, dtype=torch.long)
            
            # Create new data object
            resampled_data = Data(
                x=data.x,
                edge_index=torch.tensor(combined_edge_index, dtype=torch.long),
                edge_attr=combined_edge_attr,
                edge_labels=combined_edge_labels
            )
        else:
            # No synthetic edges needed, just update labels
            resampled_data = Data(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                edge_labels=torch.tensor(edge_labels_resampled, dtype=torch.long)
            )
        
        return resampled_data

def example_usage():
    """Example usage of the GNN wrapper with generic node approach"""
    print("GNN Training Wrapper Example (Generic)")
    print("=" * 50)
    
    # Create example data (replace with your actual data)
    np.random.seed(42)
    n_interactions = 1000
    n_nodes = 100
    
    # Create synthetic interaction data
    df = pd.DataFrame({
        'source_node': np.random.randint(0, n_nodes, n_interactions),
        'target_node': np.random.randint(0, n_nodes, n_interactions),
        'amount': np.random.exponential(100, n_interactions),
        'hour': np.random.randint(0, 24, n_interactions),
        'day_of_week': np.random.randint(0, 7, n_interactions),
        'is_fraud': np.random.choice([0, 1], n_interactions, p=[0.95, 0.05])
    })
    
    # Define features
    node_features = ['amount', 'hour', 'day_of_week']
    edge_features = ['amount', 'hour', 'day_of_week']
    
    # Create wrapper
    wrapper = GNNTrainingWrapper(random_state=42)
    
    # Build graph using source/target approach
    data = wrapper.build_graph_from_tabular(
        df=df,
        node_features=node_features,
        edge_features=edge_features,
        target_col='is_fraud',
        source_node_col='source_node',
        target_node_col='target_node'
    )
    
    # Split data
    train_data, val_data, test_data = wrapper.split_graph_data_with_sampling(data)
    
    # Train model
    model, results, best_params = wrapper.train_gnn_with_optimization(
        train_data, val_data, test_data,
        sampling_method=None,  # No sampling for this example
        n_trials=10,  # Few trials for quick example
        final_training=True
    )
    
    print("Training completed!")
    return model, results, best_params

def example_single_node_usage():
    """Example with single node ID approach (e.g., BTC wallets)"""
    print("Single Node ID Example (BTC Wallets)")
    print("=" * 50)
    
    # Create example BTC transaction data
    np.random.seed(42)
    n_transactions = 1000
    n_wallets = 50
    
    # Create synthetic BTC transaction data
    df = pd.DataFrame({
        'wallet_id': np.random.randint(0, n_wallets, n_transactions),
        'amount': np.random.exponential(0.1, n_transactions),  # BTC amounts
        'hour': np.random.randint(0, 24, n_transactions),
        'day_of_week': np.random.randint(0, 7, n_transactions),
        'is_fraud': np.random.choice([0, 1], n_transactions, p=[0.95, 0.05])
    })
    
    # Define features
    node_features = ['amount', 'hour', 'day_of_week']
    edge_features = ['amount', 'hour', 'day_of_week']
    
    # Create wrapper
    wrapper = GNNTrainingWrapper(random_state=42)
    
    # Build graph using single node ID approach
    data = wrapper.build_graph_from_tabular(
        df=df,
        node_features=node_features,
        edge_features=edge_features,
        target_col='is_fraud',
        node_id_col='wallet_id'  # Single node ID approach
    )
    
    # Split data
    train_data, val_data, test_data = wrapper.split_graph_data_with_sampling(data)
    
    # Train model
    model, results, best_params = wrapper.train_gnn_with_optimization(
        train_data, val_data, test_data,
        sampling_method=None,
        n_trials=10,
        final_training=True
    )
    
    print("BTC wallet fraud detection training completed!")
    return model, results, best_params

def example_heterogeneous_usage():
    """Example demonstrating heterogeneous node features handling"""
    print("Heterogeneous Node Features Example")
    print("=" * 50)
    
    # Create example data with different feature sets for source and target nodes
    np.random.seed(42)
    n_transactions = 2000
    n_customers = 100
    n_merchants = 50
    
    # Create synthetic transaction data with different feature sets
    df = pd.DataFrame({
        # Node identifiers
        'customer_id': np.random.randint(0, n_customers, n_transactions),
        'merchant_id': np.random.randint(0, n_merchants, n_transactions),
        
        # Source node features (customer-specific) - 5 features
        'customer_age': np.random.randint(18, 80, n_transactions),
        'customer_credit_score': np.random.randint(300, 850, n_transactions),
        'customer_account_balance': np.random.exponential(5000, n_transactions),
        'customer_risk_score': np.random.uniform(0, 1, n_transactions),
        'customer_location_risk': np.random.uniform(0, 1, n_transactions),
        
        # Target node features (merchant-specific) - 3 features
        'merchant_category': np.random.choice(['retail', 'food', 'travel', 'electronics', 'services'], n_transactions),
        'merchant_risk_score': np.random.uniform(0, 1, n_transactions),
        'merchant_transaction_volume': np.random.exponential(10000, n_transactions),
        
        # Edge features (transaction-specific)
        'amount': np.random.exponential(100, n_transactions),
        'hour': np.random.randint(0, 24, n_transactions),
        'day_of_week': np.random.randint(0, 7, n_transactions),
        
        # Target variable
        'is_fraud': np.random.choice([0, 1], n_transactions, p=[0.95, 0.05])
    })
    
    # Encode merchant category
    df['merchant_category_encoded'] = df['merchant_category'].map({
        'retail': 0, 'food': 1, 'travel': 2, 'electronics': 3, 'services': 4
    })
    
    # Define different feature sets
    source_node_features = [
        'customer_age', 'customer_credit_score', 'customer_account_balance', 
        'customer_risk_score', 'customer_location_risk'
    ]  # 5 features
    
    target_node_features = [
        'merchant_category_encoded', 'merchant_risk_score', 'merchant_transaction_volume'
    ]  # 3 features
    
    edge_features = ['amount', 'hour', 'day_of_week']
    
    print(f"Source node features ({len(source_node_features)}): {source_node_features}")
    print(f"Target node features ({len(target_node_features)}): {target_node_features}")
    print(f"Edge features ({len(edge_features)}): {edge_features}")
    
    # Create wrapper
    wrapper = GNNTrainingWrapper(random_state=42)
    
    # Build graph using heterogeneous approach
    data = wrapper.build_graph_from_tabular(
        df=df,
        node_features=source_node_features + target_node_features,  # Combined for compatibility
        edge_features=edge_features,
        target_col='is_fraud',
        source_node_col='customer_id',
        target_node_col='merchant_id',
        source_node_features=source_node_features,  # Different feature sets
        target_node_features=target_node_features
    )
    
    # Split data
    train_data, val_data, test_data = wrapper.split_graph_data_with_sampling(data)
    
    # Train model
    model, results, best_params = wrapper.train_gnn_with_optimization(
        train_data, val_data, test_data,
        sampling_method=None,
        n_trials=5,  # Few trials for quick example
        final_training=True
    )
    
    print("Heterogeneous training completed!")
    print(f"Source feature dimension: {wrapper.source_feature_dim}")
    print(f"Target feature dimension: {wrapper.target_feature_dim}")
    
    return model, results, best_params

if __name__ == '__main__':
    # Run both examples
    print("Running generic node examples...")
    example_usage()
    print("\n" + "="*50)
    example_single_node_usage()
    print("\n" + "="*50)
    example_heterogeneous_usage() 