# XGBoost vs GNN Model Comparison Pipeline

A comprehensive pipeline for comparing XGBoost and Graph Neural Network (GNN) models on various datasets, with support for differentiated node features, subgraph sampling, and automated evaluation.

## 🚀 Features

- **Generic Pipeline**: Works with any tabular dataset
- **Differentiated Node Features**: Support for different features for source and target nodes
- **Subgraph Sampling**: Efficient training and inference for large graphs
- **Hyperparameter Optimization**: Optuna-based optimization for both models
- **Comprehensive Evaluation**: ROC curves, Precision-Recall curves, metrics comparison
- **Production-Ready**: Subgraph sampling for scalable inference
- **Visualization**: Automatic generation of evaluation plots

## 📊 Feature Handling Approaches

### 1. Simple Features (Same for all nodes)
```python
# All nodes use the same features
node_features = ['amount', 'hour', 'day_of_week']
edge_features = ['amount', 'hour', 'day_of_week']

pipeline.run_full_comparison(
    df=df,
    node_features=node_features,
    edge_features=edge_features,
    target_col='is_fraud',
    source_node_col='source_node',
    target_node_col='target_node'
    # No source_node_features or target_node_features specified
)
```

### 2. Differentiated Features (Different for source and target nodes)
```python
# Source nodes (e.g., customers) have different features
source_node_features = [
    'customer_age', 'customer_credit_score', 
    'customer_account_balance', 'customer_risk_score'
]

# Target nodes (e.g., merchants) have different features
target_node_features = [
    'merchant_category_encoded', 'merchant_risk_score',
    'merchant_transaction_volume', 'merchant_location_risk'
]

# Edge features (transaction-specific)
edge_features = ['amount', 'hour', 'day_of_week', 'month', 'is_weekend']

pipeline.run_full_comparison(
    df=df,
    node_features=node_features,  # For XGBoost (combined features)
    edge_features=edge_features,
    target_col='is_fraud',
    source_node_col='customer_id',
    target_node_col='merchant_id',
    source_node_features=source_node_features,  # For GNN source nodes
    target_node_features=target_node_features   # For GNN target nodes
)
```

## 🏗️ Architecture

### Files Structure
```
├── models.py                 # GNN model definitions
├── training_pipeline.py      # XGBoost training pipeline
├── gnn_wrapper.py           # GNN training wrapper with subgraph sampling
├── run_comparison.py        # Main comparison pipeline
├── visualization.py          # Evaluation plots generation
├── example_usage.py         # Usage examples
├── requirements.txt         # Dependencies
└── README.md               # This file
```

### Key Components

1. **ModelComparisonPipeline**: Main orchestrator
2. **GenericTrainingPipeline**: XGBoost training with Optuna optimization
3. **GNNTrainingWrapper**: GNN training with subgraph sampling
4. **ModelEvaluationVisualizer**: Comprehensive evaluation plots

## 🎯 Usage Examples

### Basic Usage
```python
from run_comparison import ModelComparisonPipeline

# Create pipeline
pipeline = ModelComparisonPipeline(random_state=42)

# Run comparison
xgb_model, gnn_model, comparison = pipeline.run_full_comparison(
    df=your_dataframe,
    node_features=['feature1', 'feature2'],
    edge_features=['feature1', 'feature2'],
    target_col='is_fraud',
    source_node_col='customer_id',
    target_node_col='merchant_id',
    dataset_name="Your_Dataset"
)
```

### Credit Card Fraud Detection
```python
# Source node features (customers)
source_node_features = [
    'customer_age', 'customer_credit_score', 
    'customer_account_balance', 'customer_risk_score'
]

# Target node features (merchants)
target_node_features = [
    'merchant_category_encoded', 'merchant_risk_score',
    'merchant_transaction_volume', 'merchant_location_risk'
]

# Edge features (transactions)
edge_features = ['amount', 'hour', 'day_of_week', 'month', 'is_weekend']

# Run comparison
xgb_model, gnn_model, comparison = pipeline.run_full_comparison(
    df=df,
    node_features=source_node_features + target_node_features + edge_features,
    edge_features=edge_features,
    target_col='is_fraud',
    source_node_col='customer_id',
    target_node_col='merchant_id',
    source_node_features=source_node_features,
    target_node_features=target_node_features,
    dataset_name="Credit_Card_Fraud_Dataset"
)
```

### BTC Wallet Fraud Detection
```python
# Single node ID approach
xgb_model, gnn_model, comparison = pipeline.run_full_comparison(
    df=btc_df,
    node_features=['amount', 'hour', 'day_of_week'],
    edge_features=['amount', 'hour', 'day_of_week'],
    target_col='is_fraud',
    node_id_col='wallet_id',  # Single node ID approach
    dataset_name="BTC_Wallet_Dataset"
)
```

## 📈 Output

The pipeline generates:

1. **Model Files**: Saved in `./models/xgb_models/` and `./models/gnn_models/`
2. **Results**: JSON files in `./outputs/` with detailed metrics
3. **Evaluation Plots**: PNG files with ROC curves, Precision-Recall curves, metrics comparison, and confusion matrices
4. **Comparison Report**: Detailed comparison between XGBoost and GNN performance

### Example Output Files
```
./outputs/
├── Credit_Card_Fraud_Dataset_evaluation_20241201_143022.png
├── xgb_results_20241201_143022.json
├── gnn_results_20241201_143022.json
└── model_comparison_20241201_143022.json
```

## 🔧 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

```bash
# Run the example with differentiated features
python example_usage.py

# Run the basic comparison
python run_comparison.py
```

## 📊 Model Performance

The pipeline evaluates models using:

- **Accuracy**: Overall classification accuracy
- **ROC-AUC**: Area under the ROC curve
- **Average Precision**: Area under the Precision-Recall curve
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

## 🎨 Visualization

The pipeline automatically generates comprehensive evaluation plots:

1. **ROC Curves**: Comparison of model ROC curves with AUC scores
2. **Precision-Recall Curves**: Comparison of precision-recall performance
3. **Metrics Comparison**: Bar chart comparing all metrics
4. **Confusion Matrix**: For the best performing model

## 🔍 Subgraph Sampling

For large graphs, the pipeline uses subgraph sampling:

- **Training**: Samples k-hop subgraphs around training edges
- **Inference**: Processes edges in batches with subgraph sampling
- **Configurable**: Adjustable `max_subgraph_size` and `num_hops`
- **Fallback**: Falls back to full graph if sampling fails

## 🛠️ Customization

### Hyperparameter Optimization
```python
# Adjust optimization trials
pipeline.run_full_comparison(
    # ... other parameters
    n_trials=100,  # More trials for better optimization
    final_training=True
)
```

### Subgraph Sampling
```python
# Adjust subgraph parameters
gnn_wrapper = GNNTrainingWrapper(
    random_state=42,
    max_subgraph_size=2000,  # Larger subgraphs
    num_hops=3               # More hops
)
```

### Sampling Methods
```python
# Use SMOTE for imbalanced datasets
pipeline.run_full_comparison(
    # ... other parameters
    sampling_method='smote'  # or 'undersample' or None
)
```

## 📝 Notes

- **Feature Handling**: The `node_features` parameter is used for XGBoost (combined features) and as fallback for GNN if `source_node_features`/`target_node_features` are not specified
- **Backward Compatibility**: Existing code continues to work with the simple feature approach
- **Memory Efficiency**: Subgraph sampling makes the pipeline suitable for large graphs
- **Production Ready**: Includes inference demonstration with performance metrics

## 🤝 Contributing

Feel free to submit issues and enhancement requests! 