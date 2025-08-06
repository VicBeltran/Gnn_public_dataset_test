"""
Evaluation script for pre-trained XGBoost and GNN models
Loads models from ./models directory and performs evaluation
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score, f1_score, roc_curve
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import time

from models import ImprovedEdgeGraphSAGE, HeterogeneousEdgeGraphSAGE
from gnn_wrapper import GNNTrainingWrapper

warnings.filterwarnings('ignore')

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class ModelEvaluator:
    """Evaluates pre-trained XGBoost and GNN models"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.xgb_model = None
        self.gnn_model = None
        self.xgb_params = None
        self.gnn_params = None
        self.scaler = StandardScaler()
        
        # Create output directory
        os.makedirs('./outputs', exist_ok=True)
    
    def load_and_prepare_data(self):
        """Load and prepare data for evaluation"""
        print("Loading and preparing data...")
        
        # Load the preprocessed data
        df = pd.read_csv('./preprocessed_fraud_test.csv')
        
        # Define features (same as in run_comp_main.py)
        source_node_features = ['gender', 'street', 'city',
                              'state', 'zip', 'lat', 'long', 'city_pop', 'job']
        
        target_node_features = ['merch_lat', 'merch_long']
        
        edge_features = ['amt', 'category', 'trans_year', 'trans_month', 'trans_day', 'trans_hour', 'trans_dow']
        
        # Print feature dimensions for debugging
        print(f"Source node features: {len(source_node_features)} features")
        print(f"Target node features: {len(target_node_features)} features")
        print(f"Edge features: {len(edge_features)} features")
        
        # Combined features for XGBoost
        node_features = source_node_features + target_node_features + edge_features
        
        return df, node_features, edge_features, source_node_features, target_node_features
    
    def load_models(self, model_dir="./models"):
        """Load pre-trained models from the models directory"""
        print("Loading pre-trained models...")
        
        try:
            # Load XGBoost model
            xgb_files = [f for f in os.listdir(f"{model_dir}/xgb_models") if f.endswith('.pkl')]
            if xgb_files:
                print(f"XGBoost files: {xgb_files}")
                latest_xgb = max(xgb_files, key=lambda x: os.path.getctime(f"{model_dir}/xgb_models/{x}"))
                xgb_path = f"{model_dir}/xgb_models/{latest_xgb}"
                print(f"XGBoost path: {xgb_path}")
                
                # Load XGBoost model using xgb.Booster()
                self.xgb_model = xgb.Booster()
                self.xgb_model.load_model(xgb_path)
                
                # Load XGBoost parameters
                xgb_params_file = latest_xgb.replace('.pkl', '.json').replace('xgb_model_', 'xgb_params_')
                xgb_params_path = f"{model_dir}/xgb_models/{xgb_params_file}"
                if os.path.exists(xgb_params_path):
                    with open(xgb_params_path, 'r') as f:
                        self.xgb_params = json.load(f)
                
                print(f"✓ Loaded XGBoost model: {latest_xgb}")
            else:
                print("✗ No XGBoost model found")
            
            # Load GNN model
            gnn_files = [f for f in os.listdir(f"{model_dir}/gnn_models") if f.endswith('.pt')]
            if gnn_files:
                latest_gnn = max(gnn_files, key=lambda x: os.path.getctime(f"{model_dir}/gnn_models/{x}"))
                gnn_path = f"{model_dir}/gnn_models/{latest_gnn}"
                
                # Load GNN parameters first
                gnn_params_file = latest_gnn.replace('.pt', '.json').replace('gnn_model_', 'gnn_params_')
                gnn_params_path = f"{model_dir}/gnn_models/{gnn_params_file}"
                if os.path.exists(gnn_params_path):
                    with open(gnn_params_path, 'r') as f:
                        self.gnn_params = json.load(f)
                
                # Initialize GNN model with parameters
                if self.gnn_params:
                    # Try to load as HeterogeneousEdgeGraphSAGE first (based on error message)
                    try:
                        # Get feature dimensions from parameters or use defaults
                        # The wrapper adds 2 additional features (interaction count and fraud rate) to both source and target
                        source_feature_dim = self.gnn_params.get('source_feature_dim', 11)  # 9 original + 2 additional
                        target_feature_dim = self.gnn_params.get('target_feature_dim', 4)  # 2 original + 2 additional
                        edge_dim = self.gnn_params.get('edge_dim', 7)  # Default for edge features
                        
                        print(f"Loading Heterogeneous GNN with dimensions:")
                        print(f"  Source features: {source_feature_dim}")
                        print(f"  Target features: {target_feature_dim}")
                        print(f"  Edge features: {edge_dim}")
                        
                        self.gnn_model = HeterogeneousEdgeGraphSAGE(
                            source_feature_dim=source_feature_dim,
                            target_feature_dim=target_feature_dim,
                            hidden_channels=self.gnn_params.get('hidden_channels', 128),
                            out_channels=2,
                            edge_dim=edge_dim,
                            dropout=self.gnn_params.get('dropout', 0.1)
                        ).to(DEVICE)
                        
                        # Load model weights
                        self.gnn_model.load_state_dict(torch.load(gnn_path, map_location=DEVICE))
                        self.gnn_model.eval()
                        
                        print(f"✓ Loaded Heterogeneous GNN model: {latest_gnn}")
                        
                    except Exception as e:
                        print(f"Failed to load as Heterogeneous model: {e}")
                        print("Trying ImprovedEdgeGraphSAGE...")
                        
                        # Fallback to ImprovedEdgeGraphSAGE
                        self.gnn_model = ImprovedEdgeGraphSAGE(
                            in_channels=self.gnn_params.get('in_channels', 64),
                            hidden_channels=self.gnn_params.get('hidden_channels', 128),
                            out_channels=2,
                            edge_dim=self.gnn_params.get('edge_dim', 32),
                            dropout=self.gnn_params.get('dropout', 0.1)
                        ).to(DEVICE)
                        
                        # Load model weights
                        self.gnn_model.load_state_dict(torch.load(gnn_path, map_location=DEVICE))
                        self.gnn_model.eval()
                        
                        print(f"✓ Loaded Improved GNN model: {latest_gnn}")
                else:
                    print("✗ GNN parameters not found")
            else:
                print("✗ No GNN model found")
                
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def prepare_tabular_data(self, df, feature_cols, target_col, test_size=0.2):
        """Prepare tabular data for XGBoost evaluation"""
        print("Preparing tabular data for XGBoost evaluation...")
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        # For evaluation, use the entire dataset (no train-test split)
        # Scale features using the entire dataset
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Tabular data prepared:")
        print(f"  Full dataset: {X_scaled.shape}")
        
        return X_scaled, y
    
    def evaluate_xgb_model(self, X_test, y_test):
        """Evaluate XGBoost model"""
        if self.xgb_model is None:
            print("✗ XGBoost model not loaded")
            return None
        
        print("\n" + "="*50)
        print("EVALUATING XGBOOST MODEL")
        print("="*50)
        
        # Convert to DMatrix for XGBoost Booster
        dtest = xgb.DMatrix(X_test)
        
        # Measure inference time
        print("Measuring inference time...")
        num_samples = len(X_test)
        num_runs = 10  # Run multiple times for more accurate timing
        
        # Warm-up run
        _ = self.xgb_model.predict(dtest)
        
        # Time multiple inference runs
        inference_times = []
        for _ in range(num_runs):
            start_time = time.time()
            y_pred_proba = self.xgb_model.predict(dtest)
            end_time = time.time()
            inference_times.append(end_time - start_time)
        
        avg_inference_time = np.mean(inference_times)
        avg_time_per_sample = avg_inference_time / num_samples
        
        print(f"Average inference time: {avg_inference_time:.4f} seconds")
        print(f"Average time per sample: {avg_time_per_sample:.6f} seconds")
        print(f"Throughput: {num_samples / avg_inference_time:.1f} samples/second")
        
        # Make predictions
        y_pred = (y_pred_proba > 0.5).astype(int)
        
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
        f1 = f1_score(y_test, y_pred_optimal)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_optimal)
        
        results = {
            'model_type': 'XGBoost',
            'accuracy': accuracy,
            'auc': auc,
            'ap': ap,
            'f1_score': f1,
            'optimal_threshold': optimal_threshold,
            'confusion_matrix': cm,
            'predictions': y_pred_optimal,
            'probabilities': y_pred_proba,
            'true_labels': y_test,
            'avg_inference_time': avg_inference_time,
            'avg_time_per_sample': avg_time_per_sample,
            'throughput': num_samples / avg_inference_time
        }
        
        # Print results
        print(f"\nXGBoost Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC-AUC: {auc:.4f}")
        print(f"Average Precision: {ap:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Optimal Threshold: {optimal_threshold:.3f}")
        print(f"Average Inference Time: {avg_inference_time:.4f} seconds")
        print(f"Average Time per Sample: {avg_time_per_sample:.6f} seconds")
        print(f"Throughput: {num_samples / avg_inference_time:.1f} samples/second")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred_optimal, target_names=['Not Fraud', 'Fraud']))
        
        print(f"\nConfusion Matrix:")
        print(f"True Negatives: {cm[0,0]}")
        print(f"False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}")
        print(f"True Positives: {cm[1,1]}")
        
        return results
    
    def prepare_graph_data(self, df, source_node_col, target_node_col, 
                          source_node_features, target_node_features, edge_features, target_col):
        """Prepare graph data for GNN evaluation"""
        print("Preparing graph data for GNN evaluation...")
        
        # Create GNN wrapper to handle graph construction
        gnn_wrapper = GNNTrainingWrapper(random_state=self.random_state)
        
        # Build graph using the same method as in training
        graph_data = gnn_wrapper.build_graph_from_tabular(
            df=df,
            node_features=source_node_features + target_node_features,  # Combined for compatibility
            edge_features=edge_features,
            target_col=target_col,
            source_node_col=source_node_col,
            target_node_col=target_node_col,
            source_node_features=source_node_features,
            target_node_features=target_node_features
        )
        
        # For evaluation, use the entire graph (no splitting)
        # Create a single data object with all edges
        full_data = graph_data
        
        print(f"Graph data prepared:")
        print(f"  Full dataset: {full_data.num_edges} edges")
        
        # Store the wrapper for heterogeneous data creation
        self.gnn_wrapper = gnn_wrapper
        
        return full_data
    
    def evaluate_gnn_model(self, test_data):
        """Evaluate GNN model"""
        if self.gnn_model is None:
            print("✗ GNN model not loaded")
            return None
        
        print("\n" + "="*50)
        print("EVALUATING GNN MODEL")
        print("="*50)
        
        # Move test data to device
        test_data = test_data.to(DEVICE)
        
        # Measure inference time
        print("Measuring inference time...")
        num_samples = test_data.num_edges
        num_runs = 10  # Run multiple times for more accurate timing
        
        # Make predictions based on model type
        self.gnn_model.eval()
        
        # Warm-up run
        with torch.no_grad():
            if isinstance(self.gnn_model, HeterogeneousEdgeGraphSAGE):
                x_dict, edge_index_dict, edge_attr_dict = self.gnn_wrapper._create_heterogeneous_data(test_data, DEVICE)
                _ = self.gnn_model(x_dict, edge_index_dict, edge_attr_dict)
            else:
                if hasattr(test_data, 'num_node_features') and hasattr(test_data, 'edge_attr'):
                    if (self.gnn_model.in_channels != test_data.num_node_features or 
                        self.gnn_model.edge_dim != test_data.edge_attr.shape[1]):
                        print("Updating GNN model dimensions...")
                        self.gnn_model = ImprovedEdgeGraphSAGE(
                            in_channels=test_data.num_node_features,
                            hidden_channels=self.gnn_params.get('hidden_channels', 128),
                            out_channels=2,
                            edge_dim=test_data.edge_attr.shape[1],
                            dropout=self.gnn_params.get('dropout', 0.1)
                        ).to(DEVICE)
                _ = self.gnn_model(test_data.x, test_data.edge_index, test_data.edge_attr)
        
        # Time multiple inference runs
        inference_times = []
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                if isinstance(self.gnn_model, HeterogeneousEdgeGraphSAGE):
                    # Handle heterogeneous graph
                    print("Using heterogeneous GNN model...")
                    # Use the wrapper's heterogeneous data creation method
                    x_dict, edge_index_dict, edge_attr_dict = self.gnn_wrapper._create_heterogeneous_data(test_data, DEVICE)
                    out = self.gnn_model(x_dict, edge_index_dict, edge_attr_dict)
                else:
                    # Handle homogeneous graph
                    print("Using homogeneous GNN model...")
                    out = self.gnn_model(test_data.x, test_data.edge_index, test_data.edge_attr)
            end_time = time.time()
            inference_times.append(end_time - start_time)
        
        avg_inference_time = np.mean(inference_times)
        avg_time_per_sample = avg_inference_time / num_samples
        
        print(f"Average inference time: {avg_inference_time:.4f} seconds")
        print(f"Average time per sample: {avg_time_per_sample:.6f} seconds")
        print(f"Throughput: {num_samples / avg_inference_time:.1f} samples/second")
        
        # Final prediction for metrics
        with torch.no_grad():
            if isinstance(self.gnn_model, HeterogeneousEdgeGraphSAGE):
                x_dict, edge_index_dict, edge_attr_dict = self.gnn_wrapper._create_heterogeneous_data(test_data, DEVICE)
                out = self.gnn_model(x_dict, edge_index_dict, edge_attr_dict)
            else:
                out = self.gnn_model(test_data.x, test_data.edge_index, test_data.edge_attr)
            
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
        f1 = f1_score(y_test, y_pred_optimal)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_optimal)
        
        results = {
            'model_type': 'GNN',
            'accuracy': accuracy,
            'auc': auc,
            'ap': ap,
            'f1_score': f1,
            'optimal_threshold': optimal_threshold,
            'confusion_matrix': cm,
            'predictions': y_pred_optimal,
            'probabilities': y_pred_proba,
            'true_labels': y_test,
            'avg_inference_time': avg_inference_time,
            'avg_time_per_sample': avg_time_per_sample,
            'throughput': num_samples / avg_inference_time
        }
        
        # Print results
        print(f"\nGNN Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC-AUC: {auc:.4f}")
        print(f"Average Precision: {ap:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Optimal Threshold: {optimal_threshold:.3f}")
        print(f"Average Inference Time: {avg_inference_time:.4f} seconds")
        print(f"Average Time per Sample: {avg_time_per_sample:.6f} seconds")
        print(f"Throughput: {num_samples / avg_inference_time:.1f} samples/second")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred_optimal, target_names=['Not Fraud', 'Fraud']))
        
        print(f"\nConfusion Matrix:")
        print(f"True Negatives: {cm[0,0]}")
        print(f"False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}")
        print(f"True Positives: {cm[1,1]}")
        
        return results
    
    def compare_models(self, xgb_results, gnn_results):
        """Compare XGBoost and GNN model performance"""
        if xgb_results is None or gnn_results is None:
            print("✗ Cannot compare models - missing results")
            return None
        
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        # Create comparison table
        comparison_data = {
            'Metric': ['Accuracy', 'ROC-AUC', 'Average Precision', 'F1-Score', 'Inference Time (s)', 'Time per Sample (ms)', 'Throughput (samples/s)'],
            'XGBoost': [
                f"{xgb_results['accuracy']:.4f}",
                f"{xgb_results['auc']:.4f}",
                f"{xgb_results['ap']:.4f}",
                f"{xgb_results['f1_score']:.4f}",
                f"{xgb_results['avg_inference_time']:.4f}",
                f"{xgb_results['avg_time_per_sample']*1000:.2f}",
                f"{xgb_results['throughput']:.1f}"
            ],
            'GNN': [
                f"{gnn_results['accuracy']:.4f}",
                f"{gnn_results['auc']:.4f}",
                f"{gnn_results['ap']:.4f}",
                f"{gnn_results['f1_score']:.4f}",
                f"{gnn_results['avg_inference_time']:.4f}",
                f"{gnn_results['avg_time_per_sample']*1000:.2f}",
                f"{gnn_results['throughput']:.1f}"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\nPerformance Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Determine winner for each metric
        print("\nWinner Analysis:")
        metrics = ['accuracy', 'auc', 'ap', 'f1_score', 'avg_inference_time', 'avg_time_per_sample', 'throughput']
        metric_names = ['Accuracy', 'ROC-AUC', 'Average Precision', 'F1-Score', 'Inference Time', 'Time per Sample', 'Throughput']
        better_metrics = ['higher', 'higher', 'higher', 'higher', 'lower', 'lower', 'higher']  # Which is better
        
        for metric, name, better in zip(metrics, metric_names, better_metrics):
            xgb_val = xgb_results[metric]
            gnn_val = gnn_results[metric]
            
            if better == 'higher':
                if xgb_val > gnn_val:
                    winner = "XGBoost"
                    diff = xgb_val - gnn_val
                elif gnn_val > xgb_val:
                    winner = "GNN"
                    diff = gnn_val - xgb_val
                else:
                    winner = "Tie"
                    diff = 0
            else:  # lower is better
                if xgb_val < gnn_val:
                    winner = "XGBoost"
                    diff = gnn_val - xgb_val
                elif gnn_val < xgb_val:
                    winner = "GNN"
                    diff = xgb_val - gnn_val
                else:
                    winner = "Tie"
                    diff = 0
            
            if metric in ['avg_inference_time', 'avg_time_per_sample']:
                print(f"{name}: {winner} (diff: {diff:.6f})")
            elif metric == 'throughput':
                print(f"{name}: {winner} (diff: {diff:.1f})")
            else:
                print(f"{name}: {winner} (diff: {diff:.4f})")
        
        # Create comprehensive visualization
        self.create_comparison_plots(xgb_results, gnn_results)
        
        return comparison_df
    
    def create_comparison_plots(self, xgb_results, gnn_results):
        """Create comprehensive comparison plots similar to the reference image"""
        print("\nCreating comparison plots...")
        
        # Set up the figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('XGBoost vs GNN Model Comparison', fontsize=16, fontweight='bold')
        
        # Colors for consistency
        colors = {'XGBoost': '#FF6B35', 'GNN': '#4ECDC4'}
        
        # 1. ROC Curves Comparison (Top-Left)
        ax1 = axes[0, 0]
        self._plot_roc_curves(ax1, xgb_results, gnn_results, colors)
        
        # 2. Precision-Recall Curves Comparison (Top-Right)
        ax2 = axes[0, 1]
        self._plot_precision_recall_curves(ax2, xgb_results, gnn_results, colors)
        
        # 3. Performance Metrics Bar Chart (Bottom-Left)
        ax3 = axes[1, 0]
        self._plot_metrics_comparison(ax3, xgb_results, gnn_results, colors)
        
        # 4. Confusion Matrix for Best Model (Bottom-Right)
        ax4 = axes[1, 1]
        self._plot_best_confusion_matrix(ax4, xgb_results, gnn_results, colors)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f'./outputs/model_comparison_plots_{timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plots saved to {plot_filename}")
        
        plt.show()
    
    def _plot_roc_curves(self, ax, xgb_results, gnn_results, colors):
        """Plot ROC curves comparison"""
        # Calculate ROC curves
        xgb_fpr, xgb_tpr, _ = roc_curve(
            xgb_results['true_labels'], xgb_results['probabilities']
        )
        gnn_fpr, gnn_tpr, _ = roc_curve(
            gnn_results['true_labels'], gnn_results['probabilities']
        )
        
        # Plot curves
        ax.plot(xgb_fpr, xgb_tpr, color=colors['XGBoost'], linewidth=2, 
                label=f'XGBoost (AUC = {xgb_results["auc"]:.3f})')
        ax.plot(gnn_fpr, gnn_tpr, color=colors['GNN'], linewidth=2, 
                label=f'GNN (AUC = {gnn_results["auc"]:.3f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_precision_recall_curves(self, ax, xgb_results, gnn_results, colors):
        """Plot precision-recall curves comparison"""
        # Calculate precision-recall curves
        xgb_precision, xgb_recall, _ = precision_recall_curve(
            xgb_results['true_labels'], xgb_results['probabilities']
        )
        gnn_precision, gnn_recall, _ = precision_recall_curve(
            gnn_results['true_labels'], gnn_results['probabilities']
        )
        
        # Plot curves
        ax.plot(xgb_recall, xgb_precision, color=colors['XGBoost'], linewidth=2,
                label=f'XGBoost (AP = {xgb_results["ap"]:.3f})')
        ax.plot(gnn_recall, gnn_precision, color=colors['GNN'], linewidth=2,
                label=f'GNN (AP = {gnn_results["ap"]:.3f})')
        
        # Plot baseline
        baseline = np.mean(xgb_results['true_labels'])
        ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, 
                  label=f'Baseline (AP = {baseline:.3f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_metrics_comparison(self, ax, xgb_results, gnn_results, colors):
        """Plot performance metrics bar chart"""
        metrics = ['Accuracy', 'ROC-AUC', 'Average Precision', 'F1-Score']
        xgb_values = [
            xgb_results['accuracy'],
            xgb_results['auc'],
            xgb_results['ap'],
            xgb_results['f1_score']
        ]
        gnn_values = [
            gnn_results['accuracy'],
            gnn_results['auc'],
            gnn_results['ap'],
            gnn_results['f1_score']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, xgb_values, width, label='XGBoost', 
                       color=colors['XGBoost'], alpha=0.8)
        bars2 = ax.bar(x + width/2, gnn_values, width, label='GNN', 
                       color=colors['GNN'], alpha=0.8)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)
    
    def _plot_best_confusion_matrix(self, ax, xgb_results, gnn_results, colors):
        """Plot confusion matrix for the best performing model"""
        # Determine which model performed better (using F1-score as criterion)
        if xgb_results['f1_score'] > gnn_results['f1_score']:
            best_model = 'XGBoost'
            best_results = xgb_results
            color = colors['XGBoost']
        else:
            best_model = 'GNN'
            best_results = gnn_results
            color = colors['GNN']
        
        # Create confusion matrix
        cm = best_results['confusion_matrix']
        
        # Plot confusion matrix
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues', alpha=0.8)
        ax.figure.colorbar(im, ax=ax)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{best_model} Confusion Matrix (Best F1)')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Not Fraud', 'Fraud'])
        ax.set_yticklabels(['Not Fraud', 'Fraud'])
    
    def save_results(self, xgb_results, gnn_results, comparison_df):
        """Save evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual results
        if xgb_results:
            xgb_filename = f'./outputs/xgb_evaluation_{timestamp}.json'
            with open(xgb_filename, 'w') as f:
                # Convert numpy arrays and other types to JSON serializable
                xgb_results_json = {}
                for k, v in xgb_results.items():
                    if isinstance(v, np.ndarray):
                        xgb_results_json[k] = v.tolist()
                    elif isinstance(v, np.integer):
                        xgb_results_json[k] = int(v)
                    elif isinstance(v, np.floating):
                        xgb_results_json[k] = float(v)
                    elif isinstance(v, np.bool_):
                        xgb_results_json[k] = bool(v)
                    else:
                        xgb_results_json[k] = v
                json.dump(xgb_results_json, f, indent=4)
            print(f"✓ XGBoost results saved to {xgb_filename}")
        
        if gnn_results:
            gnn_filename = f'./outputs/gnn_evaluation_{timestamp}.json'
            with open(gnn_filename, 'w') as f:
                # Convert numpy arrays and other types to JSON serializable
                gnn_results_json = {}
                for k, v in gnn_results.items():
                    if isinstance(v, np.ndarray):
                        gnn_results_json[k] = v.tolist()
                    elif isinstance(v, np.integer):
                        gnn_results_json[k] = int(v)
                    elif isinstance(v, np.floating):
                        gnn_results_json[k] = float(v)
                    elif isinstance(v, np.bool_):
                        gnn_results_json[k] = bool(v)
                    else:
                        gnn_results_json[k] = v
                json.dump(gnn_results_json, f, indent=4)
            print(f"✓ GNN results saved to {gnn_filename}")
        
        # Save comparison
        if comparison_df is not None:
            comparison_filename = f'./outputs/model_comparison_{timestamp}.csv'
            comparison_df.to_csv(comparison_filename, index=False)
            print(f"✓ Model comparison saved to {comparison_filename}")
    
    def run_evaluation(self):
        """Run the complete evaluation pipeline"""
        print("Starting Model Evaluation Pipeline")
        print("=" * 50)
        
        # Load and prepare data
        df, node_features, edge_features, source_node_features, target_node_features = self.load_and_prepare_data()
        
        # Load pre-trained models
        self.load_models()
        
        # Evaluate XGBoost model
        X_test, y_test = self.prepare_tabular_data(
            df, node_features + edge_features, 'is_fraud'
        )
        xgb_results = self.evaluate_xgb_model(X_test, y_test)
        
        # Evaluate GNN model
        full_data = self.prepare_graph_data(
            df, 'cc_num', 'merchant', source_node_features, target_node_features, 
            edge_features, 'is_fraud'
        )
        gnn_results = self.evaluate_gnn_model(full_data)
        
        # Compare models
        comparison_df = self.compare_models(xgb_results, gnn_results)
        
        # Save results
        self.save_results(xgb_results, gnn_results, comparison_df)
        
        print("\n" + "="*50)
        print("EVALUATION COMPLETED")
        print("="*50)
        print("Check the ./outputs/ directory for detailed results.")
        
        return xgb_results, gnn_results, comparison_df

def main():
    """Main function to run the evaluation"""
    evaluator = ModelEvaluator(random_state=42)
    xgb_results, gnn_results, comparison = evaluator.run_evaluation()
    
    return xgb_results, gnn_results, comparison

if __name__ == '__main__':
    main() 