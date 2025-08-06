"""
Example usage of the training pipeline with generic node approach
"""

import pandas as pd
import numpy as np
from run_comparison import ModelComparisonPipeline

def load_and_prepare_your_data():
    """Load and prepare your data - replace with your actual data loading logic"""
    print("Loading and preparing data...")
    
    # Create example data with different features for source and target nodes
    np.random.seed(42)
    df = pd.read_csv('./preprocessed_fraud.csv')
    
    return df

def main():
    """Main function to run the complete comparison pipeline"""
    print("Starting Model Comparison Pipeline")
    print("=" * 50)
    
    # Load and prepare your data
    df = load_and_prepare_your_data()
    
    # Define features for different node types
    # Source node features (customer-specific)
    source_node_features = ['gender', 'street', 'city',
       'state', 'zip', 'lat', 'long', 'city_pop', 'job']
    
    # Target node features (merchant-specific)
    target_node_features = ['merch_lat', 'merch_long']
    
    # Edge features (transaction-specific)
    edge_features = ['amt','category', 'trans_year','trans_month', 'trans_day', 'trans_hour', 'trans_dow']
    
    # Common features (for backward compatibility and XGBoost)
    node_features = source_node_features + target_node_features + edge_features
    
    # Create pipeline
    pipeline = ModelComparisonPipeline(random_state=42)
    
    # Run full comparison with differentiated node features
    xgb_model, gnn_model, comparison = pipeline.run_full_comparison(
        df=df,
        node_features=node_features,  # For XGBoost (combined features)
        edge_features=edge_features,
        target_col='is_fraud',
        source_node_col='cc_num',
        target_node_col='merchant',
        source_node_features=source_node_features,  # For GNN source nodes
        target_node_features=target_node_features,  # For GNN target nodes
        sampling_method=None,  # No sampling for this example
        n_trials=10,  # Few trials for quick example
        final_training=True,
        dataset_name="Preprocessed_Fraud"  # Dataset name for plots
    )
    
    print("\nPipeline completed successfully!")
    print("Check the ./outputs/ directory for results and evaluation plots.")
    
    return xgb_model, gnn_model, comparison


if __name__ == '__main__':
    # Run the main example with differentiated features
    main()
