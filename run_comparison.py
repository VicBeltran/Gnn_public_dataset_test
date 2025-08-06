import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import json
from datetime import datetime
import os

from training_pipeline import GenericTrainingPipeline
from gnn_wrapper import GNNTrainingWrapper
from visualization import create_evaluation_plots_from_results
import time

warnings.filterwarnings('ignore')

class ModelComparisonPipeline:
    """Pipeline for comparing XGBoost and GNN models"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = {}
        
        # Create output directories
        os.makedirs('./outputs', exist_ok=True)
        os.makedirs('./models/xgb_models', exist_ok=True)
        os.makedirs('./models/gnn_models', exist_ok=True)
    
    def prepare_tabular_data(self, df, feature_cols, target_col, test_size=0.2, val_size=0.2):
        """Prepare tabular data for XGBoost training"""
        print("Preparing tabular data for XGBoost...")
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), 
            random_state=self.random_state, stratify=y_temp
        )
        
        print(f"Tabular data split:")
        print(f"  Train: {X_train.shape}")
        print(f"  Validation: {X_val.shape}")
        print(f"  Test: {X_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def train_xgb_model(self, df, node_features, edge_features, target_col,
                       sampling_method=None, n_trials=50, final_training=True):
        """Train XGBoost model with optimization"""
        print("\n" + "="*50)
        print("TRAINING XGBOOST MODEL")
        print("="*50)
        
        # Prepare tabular data for XGBoost (combine all features)
        feature_cols = node_features + edge_features
        X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_tabular_data(
            df, feature_cols, target_col
        )
        
        # Create XGBoost pipeline
        xgb_pipeline = GenericTrainingPipeline(
            model_type='xgb',
            sampling_method=sampling_method,
            random_state=self.random_state
        )
        
        # Train and evaluate
        model, results = xgb_pipeline.train_and_evaluate(
            X_train, y_train, X_val, y_val, X_test, y_test,
            optimize=True, n_trials=n_trials, final_training=final_training
        )
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        xgb_pipeline.save_results(results, f'./outputs/xgb_results_{timestamp}.json')
        
        self.results['xgb'] = results
        return model, results
    
    def train_gnn_model(self, df, node_features, edge_features, target_col,
                       source_node_col=None, target_node_col=None, node_id_col=None,
                       source_node_features=None, target_node_features=None,
                       sampling_method=None, n_trials=50, final_training=True):
        """Train GNN model with optimization using subgraph sampling"""
        print("\n" + "="*50)
        print("TRAINING GNN MODEL WITH SUBGRAPH SAMPLING")
        print("="*50)
        
        # Create GNN wrapper (now with subgraph sampling parameters)
        gnn_wrapper = GNNTrainingWrapper(
            random_state=self.random_state,
            max_subgraph_size=1000, # Example: Max 1000 nodes per subgraph
            num_hops=2              # Example: 2-hop neighborhood
        )
        
        # Build graph
        data = gnn_wrapper.build_graph_from_tabular(
            df=df,
            node_features=node_features,
            edge_features=edge_features,
            target_col=target_col,
            source_node_col=source_node_col,
            target_node_col=target_node_col,
            node_id_col=node_id_col,
            source_node_features=source_node_features,
            target_node_features=target_node_features
        )
        
        # Split graph data with subgraph sampling
        train_data, val_data, test_data = gnn_wrapper.split_graph_data_with_sampling(data)
        
        # Train model
        model, results, best_params = gnn_wrapper.train_gnn_with_optimization(
            train_data, val_data, test_data,
            sampling_method=sampling_method,
            n_trials=n_trials,
            final_training=final_training
        )
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gnn_wrapper.save_results(results, f'./outputs/gnn_results_{timestamp}.json')
        
        self.results['gnn'] = results
        
        # Demonstrate production inference
        if model is not None:
            try:
                gnn_wrapper.demonstrate_production_inference(model, data) # Pass the full graph for inference demo
            except Exception as e:
                print(f"Warning: Production inference demonstration failed: {e}")
                print("Continuing with pipeline...")
        
        return model, results
    
    def compare_models(self):
        """Compare XGBoost and GNN model performance"""
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        if 'xgb' not in self.results or 'gnn' not in self.results:
            print("Both models must be trained before comparison")
            return
        
        xgb_results = self.results['xgb']
        gnn_results = self.results['gnn']
        
        # Create comparison table
        comparison = {
            'metric': ['Accuracy', 'ROC-AUC', 'Average Precision', 'F1-Score'],
            'XGBoost': [
                xgb_results['accuracy'],
                xgb_results['auc'],
                xgb_results['ap'],
                xgb_results['f1_score']
            ],
            'GNN': [
                gnn_results['accuracy'],
                gnn_results['auc'],
                gnn_results['ap'],
                gnn_results['f1_score']
            ]
        }
        
        # Print comparison
        print("\nPerformance Comparison:")
        print(f"{'Metric':<20} {'XGBoost':<12} {'GNN':<12} {'Difference':<12}")
        print("-" * 56)
        
        for i, metric in enumerate(comparison['metric']):
            xgb_val = comparison['XGBoost'][i]
            gnn_val = comparison['GNN'][i]
            diff = gnn_val - xgb_val
            diff_str = f"{diff:+.4f}"
            
            print(f"{metric:<20} {xgb_val:<12.4f} {gnn_val:<12.4f} {diff_str:<12}")
        
        # Determine winner for each metric
        print("\nWinner Analysis:")
        for i, metric in enumerate(comparison['metric']):
            xgb_val = comparison['XGBoost'][i]
            gnn_val = comparison['GNN'][i]
            
            if gnn_val > xgb_val:
                winner = "GNN"
                improvement = ((gnn_val - xgb_val) / xgb_val) * 100
            else:
                winner = "XGBoost"
                improvement = ((xgb_val - gnn_val) / gnn_val) * 100
            
            print(f"{metric}: {winner} wins ({improvement:+.2f}% improvement)")
        
        # Save comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_path = f'./outputs/model_comparison_{timestamp}.json'
        
        # Convert numpy types to JSON serializable
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        comparison_data = {
            'comparison': comparison,
            'xgb_results': convert_numpy_types(xgb_results),
            'gnn_results': convert_numpy_types(gnn_results),
            'timestamp': timestamp
        }
        
        with open(comparison_path, 'w') as f:
            json.dump(comparison_data, f, indent=4)
        
        print(f"\nComparison saved to {comparison_path}")
        
        return comparison
    
    def run_full_comparison(self, df, node_features, edge_features, target_col,
                           source_node_col=None, target_node_col=None, node_id_col=None,
                           source_node_features=None, target_node_features=None,
                           sampling_method=None, n_trials=50, final_training=True,
                           dataset_name="dataset"):
        """Run full comparison between XGBoost and GNN models"""
        print("\n" + "="*60)
        print("FULL MODEL COMPARISON PIPELINE")
        print("="*60)
        
        # Train XGBoost model
        print("\nTraining XGBoost model...")
        xgb_model, xgb_results = self.train_xgb_model(
            df, node_features, edge_features, target_col, sampling_method, n_trials, final_training
        )
        
        # Train GNN model
        print("\nTraining GNN model...")
        gnn_model, gnn_results = self.train_gnn_model(
            df, node_features, edge_features, target_col,
            source_node_col, target_node_col, node_id_col,
            source_node_features, target_node_features,
            sampling_method, n_trials, final_training
        )
        
        # Compare models
        comparison = self.compare_models()
        
        # Create evaluation plots
        print("\nGenerating evaluation plots...")
        try:
            visualizer = create_evaluation_plots_from_results(
                xgb_results, gnn_results, 
                dataset_name=dataset_name,
                save_path="./outputs"
            )
            print("Evaluation plots generated successfully!")
        except Exception as e:
            print(f"Warning: Could not generate evaluation plots: {e}")
        
        return xgb_model, gnn_model, comparison

def create_example_data():
    """Create example transaction data for testing"""
    print("Creating example transaction data...")
    
    np.random.seed(42)
    n_transactions = 5000
    n_nodes = 500
    
    # Create synthetic transaction data
    df = pd.DataFrame({
        'source_node': np.random.randint(0, n_nodes, n_transactions),
        'target_node': np.random.randint(0, n_nodes, n_transactions),
        'amount': np.random.exponential(100, n_transactions),
        'hour': np.random.randint(0, 24, n_transactions),
        'day_of_week': np.random.randint(0, 7, n_transactions),
        'month': np.random.randint(1, 13, n_transactions),
        'is_weekend': np.random.choice([0, 1], n_transactions, p=[0.7, 0.3]),
        'is_fraud': np.random.choice([0, 1], n_transactions, p=[0.95, 0.05])
    })
    
    # Add some fraud patterns
    fraud_mask = df['is_fraud'] == 1
    df.loc[fraud_mask, 'amount'] = df.loc[fraud_mask, 'amount'] * np.random.uniform(2, 5, fraud_mask.sum())
    df.loc[fraud_mask, 'hour'] = np.random.choice([0, 1, 2, 3, 22, 23], fraud_mask.sum())
    
    print(f"Created dataset with {len(df)} transactions")
    print(f"Fraud rate: {df['is_fraud'].mean():.4f}")
    
    return df

def create_btc_example_data():
    """Create example BTC wallet data for testing"""
    print("Creating example BTC wallet data...")
    
    np.random.seed(42)
    n_transactions = 3000
    n_wallets = 200
    
    # Create synthetic BTC transaction data
    df = pd.DataFrame({
        'wallet_id': np.random.randint(0, n_wallets, n_transactions),
        'amount': np.random.exponential(0.1, n_transactions),  # BTC amounts
        'hour': np.random.randint(0, 24, n_transactions),
        'day_of_week': np.random.randint(0, 7, n_transactions),
        'month': np.random.randint(1, 13, n_transactions),
        'transaction_type': np.random.choice(['transfer', 'exchange', 'mining'], n_transactions),
        'is_fraud': np.random.choice([0, 1], n_transactions, p=[0.95, 0.05])
    })
    
    # Add some fraud patterns
    fraud_mask = df['is_fraud'] == 1
    df.loc[fraud_mask, 'amount'] = df.loc[fraud_mask, 'amount'] * np.random.uniform(5, 20, fraud_mask.sum())
    df.loc[fraud_mask, 'transaction_type'] = 'transfer'  # Most fraud is transfers
    
    # Encode transaction type
    df['transaction_type_encoded'] = df['transaction_type'].map({'transfer': 0, 'exchange': 1, 'mining': 2})
    
    print(f"Created BTC dataset with {len(df)} transactions")
    print(f"Fraud rate: {df['is_fraud'].mean():.4f}")
    
    return df