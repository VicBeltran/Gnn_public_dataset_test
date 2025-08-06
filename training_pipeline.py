import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import warnings
import json
from datetime import datetime
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from models import ImprovedEdgeGraphSAGE
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class FocalLoss(torch.nn.Module):
    """Focal Loss for heavily imbalanced classification with improved numerical stability"""
    def __init__(self, alpha=1, gamma=2, pos_weight=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Ensure inputs are in the right range for numerical stability
        inputs = torch.clamp(inputs, min=-100, max=100)
        
        # Use log_softmax for better numerical stability
        log_probs = torch.nn.functional.log_softmax(inputs, dim=1)
        
        # Gather the log probabilities for the target classes
        batch_size = targets.size(0)
        target_log_probs = log_probs[torch.arange(batch_size), targets]
        
        # Calculate pt (probability of target class)
        pt = torch.exp(target_log_probs)
        pt = torch.clamp(pt, min=1e-7, max=1.0 - 1e-7)  # Clamp to avoid log(0) or log(1)
        
        # Calculate focal loss
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = -self.alpha * focal_weight * target_log_probs
        
        # Apply class weights if provided
        if self.pos_weight is not None:
            # Create weight tensor based on target classes
            weights = torch.ones_like(targets, dtype=torch.float)
            weights[targets == 1] = self.pos_weight[1]
            weights[targets == 0] = self.pos_weight[0]
            focal_loss = focal_loss * weights
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class GenericTrainingPipeline:
    """Generic training pipeline for both XGBoost and GNN models"""
    
    def __init__(self, model_type='gnn', sampling_method=None, random_state=42):
        """
        Initialize the training pipeline
        
        Args:
            model_type (str): 'gnn' or 'xgb'
            sampling_method (str): 'smote', 'undersample', or None
            random_state (int): Random seed for reproducibility
        """
        self.model_type = model_type.lower()
        self.sampling_method = sampling_method
        self.random_state = random_state
        self.best_params = None
        self.best_model = None
        self.scaler = StandardScaler()
        
        # Create model directories
        os.makedirs('./models/gnn_models', exist_ok=True)
        os.makedirs('./models/xgb_models', exist_ok=True)
        os.makedirs('./outputs', exist_ok=True)
        
    def build_graph_from_data(self, node_features, edge_features, edge_index, target):
        """Build PyTorch Geometric Data object from tabular data"""
        # Convert to tensors
        node_features = torch.tensor(node_features, dtype=torch.float)
        edge_features = torch.tensor(edge_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)
        
        # Create Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            edge_labels=target
        )
        
        # Sort for LSTM compatibility
        data = data.sort(sort_by_row=False)
        
        return data
    
    def apply_sampling(self, X, y, method='smote'):
        """Apply sampling technique to balance classes"""
        if method == 'smote':
            smote = SMOTE(random_state=self.random_state)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            print(f"SMOTE applied - Original shape: {X.shape}, Resampled shape: {X_resampled.shape}")
            return X_resampled, y_resampled
        elif method == 'undersample':
            undersampler = RandomUnderSampler(random_state=self.random_state)
            X_resampled, y_resampled = undersampler.fit_resample(X, y)
            print(f"Undersampling applied - Original shape: {X.shape}, Resampled shape: {X_resampled.shape}")
            return X_resampled, y_resampled
        else:
            return X, y
    
    def objective_xgb(self, trial, X_train, y_train, X_val, y_val):
        """Optuna objective function for XGBoost hyperparameter optimization"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),
            'random_state': self.random_state,
            'eval_metric': 'auc',
            'early_stopping_rounds': 50,
            'verbose': 0
        }
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Predict and evaluate
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)
        
        return auc
    
    def objective_gnn(self, trial, data_train, data_val):
        """Optuna objective function for GNN hyperparameter optimization"""
        params = {
            'hidden_channels': trial.suggest_int('hidden_channels', 64, 256),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
            'alpha': trial.suggest_float('alpha', 0.5, 2.0),
            'gamma': trial.suggest_float('gamma', 1.0, 4.0),
        }
        
        # Initialize model
        model = ImprovedEdgeGraphSAGE(
            in_channels=data_train.num_node_features,
            hidden_channels=params['hidden_channels'],
            out_channels=2,  # Binary classification
            edge_dim=data_train.edge_attr.shape[1],
            dropout=params['dropout']
        ).to(DEVICE)
        
        # Loss function with focal loss
        pos_weight = torch.tensor([1.0, 5.0]).to(DEVICE)  # Adjust based on class imbalance
        criterion = FocalLoss(
            alpha=params['alpha'],
            gamma=params['gamma'],
            pos_weight=pos_weight
        ).to(DEVICE)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['lr'],
            weight_decay=params['weight_decay']
        )
        
        # Training loop
        best_val_auc = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(50):  # Shorter training for optimization
            model.train()
            optimizer.zero_grad()
            
            out = model(data_train.x, data_train.edge_index, data_train.edge_attr)
            loss = criterion(out, data_train.edge_labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_out = model(data_val.x, data_val.edge_index, data_val.edge_attr)
                val_predictions = torch.softmax(val_out, dim=1)
                val_probs = val_predictions[:, 1].cpu().numpy()
                val_auc = roc_auc_score(data_val.edge_labels.cpu().numpy(), val_probs)
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    break
        
        return best_val_auc
    
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, n_trials=50):
        """Optimize hyperparameters using Optuna"""
        print(f"Optimizing hyperparameters for {self.model_type.upper()} model...")
        
        if self.model_type == 'xgb':
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=self.random_state)
            )
            study.optimize(
                lambda trial: self.objective_xgb(trial, X_train, y_train, X_val, y_val),
                n_trials=n_trials
            )
            
            self.best_params = study.best_params
            print(f"Best XGBoost parameters: {self.best_params}")
            print(f"Best validation AUC: {study.best_value:.4f}")
            
        elif self.model_type == 'gnn':
            # For GNN, we need to build the graph data
            # This assumes edge_features and edge_index are passed separately
            # You'll need to modify this based on your data structure
            print("GNN optimization requires graph data structure")
            return None
        
        return study
    
    def train_xgb_model(self, X_train, y_train, X_val, y_val, params=None):
        """Train XGBoost model with optimal parameters"""
        if params is None:
            params = self.best_params
        
        if params is None:
            # Default parameters if no optimization was done
            params = {
                'n_estimators': 500,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 1.0,
                'reg_lambda': 1.0,
                'scale_pos_weight': 5.0,
                'random_state': self.random_state,
                'eval_metric': 'auc',
                'early_stopping_rounds': 50,
                'verbose': 0
            }
        
        print("Training XGBoost model with parameters:", params)
        
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f'./models/xgb_models/xgb_model_{timestamp}.pkl'
        model.save_model(model_path)
        
        # Save parameters
        params_path = f'./models/xgb_models/xgb_params_{timestamp}.json'
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=4)
        
        self.best_model = model
        print(f"XGBoost model saved to {model_path}")
        
        return model
    
    def train_gnn_model(self, data_train, data_val, params=None):
        """Train GNN model with optimal parameters"""
        if params is None:
            params = {
                'hidden_channels': 128,
                'dropout': 0.1,
                'lr': 0.001,
                'weight_decay': 0.001,
                'alpha': 1.0,
                'gamma': 2.0,
            }
        
        print("Training GNN model with parameters:", params)
        
        # Initialize model
        model = ImprovedEdgeGraphSAGE(
            in_channels=data_train.num_node_features,
            hidden_channels=params['hidden_channels'],
            out_channels=2,
            edge_dim=data_train.edge_attr.shape[1],
            dropout=params['dropout']
        ).to(DEVICE)
        
        # Loss function
        pos_weight = torch.tensor([1.0, 5.0]).to(DEVICE)
        criterion = FocalLoss(
            alpha=params['alpha'],
            gamma=params['gamma'],
            pos_weight=pos_weight
        ).to(DEVICE)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['lr'],
            weight_decay=params['weight_decay']
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        training_losses = []
        validation_losses = []
        
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            
            out = model(data_train.x, data_train.edge_index, data_train.edge_attr)
            loss = criterion(out, data_train.edge_labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_out = model(data_val.x, data_val.edge_index, data_val.edge_attr)
                val_loss = criterion(val_out, data_val.edge_labels)
                
                # Calculate validation AUC
                val_predictions = torch.softmax(val_out, dim=1)
                val_probs = val_predictions[:, 1].cpu().numpy()
                val_auc = roc_auc_score(data_val.edge_labels.cpu().numpy(), val_probs)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Track losses
            training_losses.append(loss.item())
            validation_losses.append(val_loss.item())
            
            # Early stopping
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
                # Save best model
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = f'./models/gnn_models/gnn_model_{timestamp}.pt'
                torch.save(model.state_dict(), model_path)
                
                # Save parameters
                params_path = f'./models/gnn_models/gnn_params_{timestamp}.json'
                with open(params_path, 'w') as f:
                    json.dump(params, f, indent=4)
            else:
                patience_counter += 1
            
            if epoch % 20 == 0:
                print(f'Epoch {epoch+1}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}, Val AUC: {val_auc:.4f}')
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        model.load_state_dict(torch.load(model_path))
        self.best_model = model
        print(f"GNN model saved to {model_path}")
        
        return model, training_losses, validation_losses
    
    def evaluate_model(self, model, X_test, y_test, model_type=None):
        """Evaluate model performance"""
        if model_type is None:
            model_type = self.model_type
        
        if model_type == 'xgb':
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
        elif model_type == 'gnn':
            model.eval()
            with torch.no_grad():
                out = model(X_test.x, X_test.edge_index, X_test.edge_attr)
                predictions = torch.softmax(out, dim=1)
                y_pred_proba = predictions[:, 1].cpu().numpy()
                y_pred = torch.argmax(predictions, dim=1).cpu().numpy()
                y_test = X_test.edge_labels.cpu().numpy()
        
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
        print(f"\n{model_type.upper()} Model Performance:")
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
    
    def train_and_evaluate(self, X_train, y_train, X_val, y_val, X_test, y_test, 
                          optimize=True, n_trials=50, final_training=True):
        """
        Complete training and evaluation pipeline
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            optimize (bool): Whether to optimize hyperparameters
            n_trials (int): Number of optimization trials
            final_training (bool): Whether to do final training with best params
        """
        print(f"Starting {self.model_type.upper()} training pipeline...")
        
        # Apply sampling if specified
        if self.sampling_method:
            X_train_resampled, y_train_resampled = self.apply_sampling(X_train, y_train, self.sampling_method)
        else:
            X_train_resampled, y_train_resampled = X_train, y_train
        
        if self.model_type == 'xgb':
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train_resampled)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)
            
            if optimize:
                # Optimize hyperparameters
                study = self.optimize_hyperparameters(X_train_scaled, y_train_resampled, 
                                                   X_val_scaled, y_val, n_trials)
            
            if final_training:
                # Train final model
                model = self.train_xgb_model(X_train_scaled, y_train_resampled, 
                                           X_val_scaled, y_val)
                
                # Evaluate
                results = self.evaluate_model(model, X_test_scaled, y_test, 'xgb')
                
                return model, results
        
        elif self.model_type == 'gnn':
            # For GNN, we assume the data is already in graph format
            # You'll need to modify this based on your specific data structure
            print("GNN training requires graph data structure")
            return None, None
    
    def save_results(self, results, filename=None):
        """Save evaluation results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'./outputs/{self.model_type}_results_{timestamp}.json'
        
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

def main():
    """Example usage of the training pipeline"""
    print("Generic Training Pipeline Example")
    print("=" * 50)
    
    # Example data (replace with your actual data)
    # For XGBoost
    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 2, 1000)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # XGBoost pipeline
    print("\nTraining XGBoost model...")
    xgb_pipeline = GenericTrainingPipeline(model_type='xgb', sampling_method='smote')
    xgb_model, xgb_results = xgb_pipeline.train_and_evaluate(
        X_train, y_train, X_val, y_val, X_test, y_test,
        optimize=True, n_trials=20, final_training=True
    )
    
    # Save XGBoost results
    xgb_pipeline.save_results(xgb_results, './outputs/xgb_results.json')
    
    print("\nTraining pipeline completed!")

if __name__ == '__main__':
    main() 