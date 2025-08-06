# Model Evaluation Script

This script (`evaluate_models.py`) provides a standalone evaluation pipeline for pre-trained XGBoost and GNN models without requiring the full training pipeline.

## Features

- **Loads pre-trained models** from `./models/xgb_models/` and `./models/gnn_models/` directories
- **Evaluates both XGBoost and GNN models** on the same test data
- **Performs comprehensive evaluation** including accuracy, ROC-AUC, Average Precision, and F1-Score
- **Compares model performance** side-by-side
- **Saves detailed results** to JSON files and comparison tables to CSV

## Usage

```bash
python evaluate_models.py
```

## Requirements

- Pre-trained models must exist in the `./models/` directory:
  - XGBoost models: `./models/xgb_models/xgb_model_*.pkl`
  - GNN models: `./models/gnn_models/gnn_model_*.pt`
  - Parameter files: `./models/xgb_models/xgb_params_*.json` and `./models/gnn_models/gnn_params_*.json`

## Output

The script generates:

1. **Console output** with detailed evaluation metrics for each model
2. **JSON files** with complete evaluation results:
   - `./outputs/xgb_evaluation_YYYYMMDD_HHMMSS.json`
   - `./outputs/gnn_evaluation_YYYYMMDD_HHMMSS.json`
3. **CSV comparison table**:
   - `./outputs/model_comparison_YYYYMMDD_HHMMSS.csv`

## Model Loading

- **XGBoost**: Uses `xgb.Booster()` to load models saved with `model.save_model()`
- **GNN**: Automatically detects model type (HeterogeneousEdgeGraphSAGE or ImprovedEdgeGraphSAGE) and loads with correct feature dimensions

## Data Processing

- Uses the same data preprocessing as the training pipeline
- Handles heterogeneous graph structures for GNN evaluation
- Maintains consistent train/test splits for fair comparison

## Example Results

```
Performance Comparison:
           Metric XGBoost    GNN
         Accuracy  0.9941 0.9389
          ROC-AUC  0.9990 0.9808
Average Precision  0.9846 0.7793
         F1-Score  0.9406 0.8098

Winner Analysis:
Accuracy: XGBoost (diff: 0.0552)
ROC-AUC: XGBoost (diff: 0.0182)
Average Precision: XGBoost (diff: 0.2053)
F1-Score: XGBoost (diff: 0.1308)
```

## Troubleshooting

- **Model loading errors**: Ensure models exist in the correct directories with matching parameter files
- **Feature dimension mismatches**: The script automatically handles different feature dimensions for heterogeneous graphs
- **JSON serialization errors**: Fixed to handle numpy data types properly

## Dependencies

- pandas, numpy, torch, torch_geometric
- xgboost, scikit-learn
- matplotlib, seaborn (for visualization)
- All dependencies from the main project 