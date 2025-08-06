import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import os
from datetime import datetime

class ModelEvaluationVisualizer:
    """Generate comprehensive evaluation plots for model comparison"""
    
    def __init__(self, dataset_name="dataset"):
        self.dataset_name = dataset_name
        self.results = {}
        
    def add_model_results(self, model_name, results):
        """Add results for a specific model"""
        self.results[model_name] = results
        
    def create_evaluation_plots(self, save_path="./outputs"):
        """Create comprehensive evaluation plots"""
        if len(self.results) < 2:
            print("Need at least 2 models for comparison")
            return
            
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Model Evaluation Comparison - {self.dataset_name}', fontsize=16, fontweight='bold')
        
        # 1. ROC Curves
        self._plot_roc_curves(ax1)
        
        # 2. Precision-Recall Curves
        self._plot_pr_curves(ax2)
        
        # 3. Performance Metrics Comparison
        self._plot_metrics_comparison(ax3)
        
        # 4. Confusion Matrix for Best Model
        self._plot_best_confusion_matrix(ax4)
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Save with dataset name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.dataset_name}_evaluation_{timestamp}.png"
        filepath = os.path.join(save_path, filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Evaluation plots saved to: {filepath}")
        
        plt.show()
        
    def _plot_roc_curves(self, ax):
        """Plot ROC curves for all models"""
        ax.set_title('ROC Curves Comparison', fontweight='bold')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        
        colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']
        
        for i, (model_name, results) in enumerate(self.results.items()):
            if 'probabilities' in results and 'true_labels' in results:
                y_true = results['true_labels']
                y_proba = results['probabilities']
                
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                auc_score = results.get('auc', 0)
                
                color = colors[i % len(colors)]
                ax.plot(fpr, tpr, color=color, linewidth=2, 
                       label=f'{model_name} (AUC: {auc_score:.3f})')
        
        # Add random classifier baseline
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_pr_curves(self, ax):
        """Plot Precision-Recall curves for all models"""
        ax.set_title('Precision-Recall Curves Comparison', fontweight='bold')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        
        colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']
        
        for i, (model_name, results) in enumerate(self.results.items()):
            if 'probabilities' in results and 'true_labels' in results:
                y_true = results['true_labels']
                y_proba = results['probabilities']
                
                precision, recall, _ = precision_recall_curve(y_true, y_proba)
                ap_score = results.get('ap', 0)
                
                color = colors[i % len(colors)]
                ax.plot(recall, precision, color=color, linewidth=2,
                       label=f'{model_name} (AP: {ap_score:.3f})')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_metrics_comparison(self, ax):
        """Plot bar chart comparing performance metrics"""
        ax.set_title('Performance Metrics Comparison', fontweight='bold')
        ax.set_ylabel('Score')
        
        metrics = ['accuracy', 'auc', 'f1_score', 'precision', 'recall']
        metric_labels = ['Accuracy', 'AUC', 'F1-Score', 'Precision', 'Recall']
        
        # Prepare data for plotting
        model_names = list(self.results.keys())
        x = np.arange(len(metrics))
        width = 0.8 / len(model_names)
        
        colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']
        
        for i, model_name in enumerate(model_names):
            results = self.results[model_name]
            scores = []
            
            for metric in metrics:
                score = results.get(metric, 0)
                scores.append(score)
            
            color = colors[i % len(colors)]
            ax.bar(x + i * width, scores, width, label=model_name, color=color, alpha=0.8)
        
        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels(metric_labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
        
    def _plot_best_confusion_matrix(self, ax):
        """Plot confusion matrix for the best performing model"""
        # Find best model based on F1 score
        best_model = None
        best_f1 = -1
        
        for model_name, results in self.results.items():
            f1_score = results.get('f1_score', 0)
            if f1_score > best_f1:
                best_f1 = f1_score
                best_model = model_name
        
        if best_model is None:
            ax.text(0.5, 0.5, 'No valid results found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Confusion Matrix - No Data')
            return
        
        results = self.results[best_model]
        cm = results.get('confusion_matrix')
        
        if cm is not None:
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Predicted 0', 'Predicted 1'],
                       yticklabels=['Actual 0', 'Actual 1'])
            
            ax.set_title(f'{best_model} Confusion Matrix (Best F1)', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No confusion matrix available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{best_model} Confusion Matrix', fontweight='bold')

def create_evaluation_plots_from_results(xgb_results, gnn_results, dataset_name="dataset", save_path="./outputs"):
    """Convenience function to create evaluation plots from XGBoost and GNN results"""
    
    # Create visualizer
    visualizer = ModelEvaluationVisualizer(dataset_name)
    
    # Add model results
    if xgb_results:
        visualizer.add_model_results("XGBoost", xgb_results)
    
    if gnn_results:
        visualizer.add_model_results("GNN", gnn_results)
    
    # Create and save plots
    visualizer.create_evaluation_plots(save_path)
    
    return visualizer

# Example usage
if __name__ == "__main__":
    # Example with synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic results for demonstration
    xgb_results = {
        'accuracy': 0.83,
        'auc': 0.59,
        'f1_score': 0.33,
        'precision': 0.22,
        'recall': 0.60,
        'ap': 0.253,
        'probabilities': np.random.random(n_samples),
        'true_labels': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'confusion_matrix': np.array([[800, 100], [80, 20]])
    }
    
    gnn_results = {
        'accuracy': 0.87,
        'auc': 0.944,
        'f1_score': 0.71,
        'precision': 0.62,
        'recall': 0.89,
        'ap': 0.800,
        'probabilities': np.random.random(n_samples),
        'true_labels': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'confusion_matrix': np.array([[3567, 401], [144, 649]])
    }
    
    # Create evaluation plots
    visualizer = create_evaluation_plots_from_results(
        xgb_results, gnn_results, 
        dataset_name="Example_Dataset"
    ) 