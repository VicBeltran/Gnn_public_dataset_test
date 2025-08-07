# GNN Inference Explorer - Streamlit App

An interactive web application for exploring Graph Neural Network (GNN) inference with subgraph visualization and real-time inference capabilities.

## Features

- **Interactive Graph Visualization**: Explore 2-hop neighborhood subgraphs around selected edges
- **Edge Selection**: Choose any edge by its index from the full graph
- **Real-time Inference**: Perform GNN inference on selected edges with subgraph sampling
- **Property Inspection**: View detailed properties of nodes and edges
- **Fraud Detection Focus**: Specifically designed for fraud detection in transaction networks

## Quick Start

### Prerequisites

1. Ensure you have the required data file:
   ```
   preprocessed_fraud_test.csv
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

#### Option 1: Using the provided script
```bash
python run_streamlit_app.py
```

#### Option 2: Direct Streamlit command
```bash
streamlit run streamlit_inference_app.py
```

The app will open in your default web browser at `http://localhost:8501`.

## How to Use

### 1. Edge Selection
- In the sidebar, you'll see the range of available edge indices
- Enter an edge index in the number input field
- Click "🔍 Get Edge" to load the subgraph around that edge

### 2. Graph Visualization
- The main area shows an interactive Plotly graph
- **Yellow lines**: Your selected edge (highlighted)
- **Red lines**: Fraudulent transactions
- **Gray lines**: Normal transactions
- **Blue nodes**: Customer nodes
- **Green nodes**: Merchant nodes

### 3. Subgraph Statistics
- View key metrics about the sampled subgraph:
  - Number of nodes
  - Number of edges
  - Fraud rate in the subgraph

### 4. Inference
- Click "🚀 Perform Inference" to run the GNN model
- View the fraud probability score on a gauge chart
- See detailed inference results and messages

### 5. Information Panel
- View properties of the selected edge
- See edge features and metadata
- Follow step-by-step instructions

## Graph Structure

The app works with a bipartite graph where:
- **Source nodes**: Customers (credit card holders)
- **Target nodes**: Merchants
- **Edges**: Transactions between customers and merchants
- **Edge features**: Transaction characteristics (amount, category, time, etc.)
- **Node features**: Customer and merchant characteristics

## Subgraph Sampling

The app uses 2-hop neighborhood sampling for inference:
1. Selects the target edge
2. Samples all nodes within 2 hops of the edge endpoints
3. Includes all edges between these nodes
4. Limits subgraph size to 500 nodes for performance

## Inference Process

1. **Subgraph Sampling**: Extract relevant neighborhood around target edge
2. **Feature Processing**: Prepare node and edge features
3. **Model Inference**: Run GNN model on the subgraph
4. **Prediction Extraction**: Extract fraud probability for the target edge
5. **Result Visualization**: Display results with interactive charts

## Technical Details

### Data Loading
- Uses `GraphDebugExplorer` to load and preprocess data
- Caches graph data for performance
- Handles both homogeneous and heterogeneous graphs

### Visualization
- Built with Plotly for interactive graphs
- Spring layout for node positioning
- Color-coded edges and nodes
- Hover information for detailed inspection

### Inference
- Subgraph sampling for scalability
- Batch processing for multiple edges
- Real-time prediction scores
- Gauge visualization for fraud probability

## File Structure

```
├── streamlit_inference_app.py    # Main Streamlit application
├── run_streamlit_app.py         # Script to run the app
├── graph_debug_explorer.py      # Graph loading and exploration
├── gnn_wrapper.py               # GNN training and inference
├── models.py                    # GNN model definitions
└── preprocessed_fraud_test.csv  # Required data file
```

## Customization

### Adding Real Model Inference
To use a trained model instead of the demonstration:

1. Load your trained model in `perform_inference_on_edge()`
2. Replace the simulated prediction with actual model output
3. Update the inference logic to handle your specific model architecture

### Modifying Graph Visualization
- Adjust colors, sizes, and layout in `create_interactive_graph_plot()`
- Add more interactive features using Plotly callbacks
- Customize hover information and legends

### Extending Features
- Add node selection and property viewing
- Implement edge filtering and search
- Add model comparison capabilities
- Include feature importance visualization

## Troubleshooting

### Common Issues

1. **Data file not found**: Ensure `preprocessed_fraud_test.csv` is in the current directory
2. **Memory issues**: Reduce `max_nodes` parameter in subgraph sampling
3. **Slow loading**: The first run may take longer due to data preprocessing
4. **Visualization errors**: Check that all required packages are installed

### Performance Tips

- Use smaller subgraph sizes for faster inference
- Cache frequently accessed data
- Optimize graph layout parameters
- Consider using GPU acceleration for large graphs

## Future Enhancements

- [ ] Real model loading and inference
- [ ] Multiple edge selection
- [ ] Advanced filtering and search
- [ ] Model comparison tools
- [ ] Export capabilities
- [ ] Batch inference interface
- [ ] Real-time data streaming
- [ ] Advanced analytics dashboard

## Contributing

Feel free to extend the app with additional features:
1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Submit a pull request

## License

This project is part of the GNN evaluation framework for fraud detection. 