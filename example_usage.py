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
    n_transactions = 5000
    n_customers = 300
    n_merchants = 200
    
    # Create synthetic transaction data with differentiated node features
    df = pd.DataFrame({
        # Node identifiers
        'customer_id': np.random.randint(0, n_customers, n_transactions),
        'merchant_id': np.random.randint(0, n_merchants, n_transactions),
        
        # Source node features (customer-specific)
        'customer_age': np.random.randint(18, 80, n_transactions),
        'customer_credit_score': np.random.randint(300, 850, n_transactions),
        'customer_account_balance': np.random.exponential(5000, n_transactions),
        'customer_risk_score': np.random.uniform(0, 1, n_transactions),
        
        # Target node features (merchant-specific)
        'merchant_category': np.random.choice(['retail', 'food', 'travel', 'electronics', 'services'], n_transactions),
        'merchant_risk_score': np.random.uniform(0, 1, n_transactions),
        'merchant_transaction_volume': np.random.exponential(10000, n_transactions),
        'merchant_location_risk': np.random.uniform(0, 1, n_transactions),
        
        # Edge features (transaction-specific)
        'amount': np.random.exponential(100, n_transactions),
        'hour': np.random.randint(0, 24, n_transactions),
        'day_of_week': np.random.randint(0, 7, n_transactions),
        'month': np.random.randint(1, 13, n_transactions),
        'is_weekend': np.random.choice([0, 1], n_transactions, p=[0.7, 0.3]),
        
        # Target variable
        'is_fraud': np.random.choice([0, 1], n_transactions, p=[0.95, 0.05])
    })
    
    # Add some fraud patterns
    fraud_mask = df['is_fraud'] == 1
    df.loc[fraud_mask, 'amount'] = df.loc[fraud_mask, 'amount'] * np.random.uniform(2, 5, fraud_mask.sum())
    df.loc[fraud_mask, 'hour'] = np.random.choice([0, 1, 2, 3, 22, 23], fraud_mask.sum())
    
    # Encode merchant category
    df['merchant_category_encoded'] = df['merchant_category'].map({
        'retail': 0, 'food': 1, 'travel': 2, 'electronics': 3, 'services': 4
    })
    
    print(f"Created dataset with {len(df)} transactions")
    print(f"Fraud rate: {df['is_fraud'].mean():.4f}")
    print(f"Number of customers: {df['customer_id'].nunique()}")
    print(f"Number of merchants: {df['merchant_id'].nunique()}")
    
    return df

def main():
    """Main function to run the complete comparison pipeline"""
    print("Starting Model Comparison Pipeline")
    print("=" * 50)
    
    # Load and prepare your data
    df = load_and_prepare_your_data()
    
    # Define features for different node types
    # Source node features (customer-specific)
    source_node_features = [
        'customer_age', 'customer_credit_score', 'customer_account_balance', 'customer_risk_score'
    ]
    
    # Target node features (merchant-specific)
    target_node_features = [
        'merchant_category_encoded', 'merchant_risk_score', 
        'merchant_transaction_volume', 'merchant_location_risk'
    ]
    
    # Edge features (transaction-specific)
    edge_features = ['amount', 'hour', 'day_of_week', 'month', 'is_weekend']
    
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
        source_node_col='customer_id',
        target_node_col='merchant_id',
        source_node_features=source_node_features,  # For GNN source nodes
        target_node_features=target_node_features,  # For GNN target nodes
        sampling_method=None,  # No sampling for this example
        n_trials=10,  # Few trials for quick example
        final_training=True,
        dataset_name="Credit_Card_Fraud_Dataset"  # Dataset name for plots
    )
    
    print("\nPipeline completed successfully!")
    print("Check the ./outputs/ directory for results and evaluation plots.")
    
    return xgb_model, gnn_model, comparison

def example_simple_features():
    """Example with simple features (same features for all nodes)"""
    print("\n" + "="*50)
    print("EXAMPLE: Simple Features (Same for all nodes)")
    print("="*50)
    
    # Create simple example data
    np.random.seed(42)
    n_transactions = 3000
    n_nodes = 200
    
    df = pd.DataFrame({
        'source_node': np.random.randint(0, n_nodes, n_transactions),
        'target_node': np.random.randint(0, n_nodes, n_transactions),
        'amount': np.random.exponential(100, n_transactions),
        'hour': np.random.randint(0, 24, n_transactions),
        'day_of_week': np.random.randint(0, 7, n_transactions),
        'is_fraud': np.random.choice([0, 1], n_transactions, p=[0.95, 0.05])
    })
    
    # Define simple features (same for all nodes)
    node_features = ['amount', 'hour', 'day_of_week']
    edge_features = ['amount', 'hour', 'day_of_week']
    
    # Create pipeline
    pipeline = ModelComparisonPipeline(random_state=42)
    
    # Run comparison with simple features
    xgb_model, gnn_model, comparison = pipeline.run_full_comparison(
        df=df,
        node_features=node_features,
        edge_features=edge_features,
        target_col='is_fraud',
        source_node_col='source_node',
        target_node_col='target_node',
        # No source_node_features or target_node_features - will use node_features for both
        sampling_method=None,
        n_trials=5,
        final_training=True,
        dataset_name="Simple_Features_Dataset"
    )
    
    return xgb_model, gnn_model, comparison

if __name__ == '__main__':
    # Run the main example with differentiated features
    main()
    
    # Run the simple features example
    example_simple_features() 