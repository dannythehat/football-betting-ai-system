"""
Deep Neural Network
Multi-layer perceptron for feature interaction modeling
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Any
import pickle
import os

from .config import DNN_CONFIG, TRAINING_CONFIG


class DeepNeuralNet(nn.Module):
    """Deep neural network architecture"""
    
    def __init__(self, input_size: int, hidden_layers: List[int], 
                 dropout: float, batch_normalization: bool):
        super(DeepNeuralNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization
            if batch_normalization:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            layers.append(nn.ReLU())
            
            # Dropout
            layers.append(nn.Dropout(dropout))
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class DeepNeuralNetwork:
    """
    Deep neural network predictor
    Captures complex non-linear feature interactions
    """
    
    def __init__(self, market: str, input_size: int):
        """
        Initialize DNN predictor
        
        Args:
            market: Target market ('goals', 'cards', 'corners', 'btts')
            input_size: Number of input features
        """
        self.market = market
        self.input_size = input_size
        self.config = DNN_CONFIG
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and TRAINING_CONFIG['use_gpu'] else 'cpu')
        
        self.model = DeepNeuralNet(
            input_size=input_size,
            hidden_layers=self.config['hidden_layers'],
            dropout=self.config['dropout'],
            batch_normalization=self.config['batch_normalization']
        ).to(self.device)
        
        self.is_trained = False
        self.best_loss = float('inf')
        self.feature_names = []
    
    def train(self, X: pd.DataFrame, y: pd.Series,
              validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None) -> Dict[str, Any]:
        """
        Train DNN model
        
        Args:
            X: Training features
            y: Training labels
            validation_data: Optional (X_val, y_val) tuple
            
        Returns:
            Training metrics
        """
        self.feature_names = list(X.columns)
        
        # Convert to tensors
        X_train = torch.FloatTensor(X.values)
        y_train = torch.FloatTensor(y.values)
        
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val = torch.FloatTensor(X_val.values)
            y_val = torch.FloatTensor(y_val.values)
        else:
            # Split validation
            split_idx = int(len(X_train) * 0.8)
            X_val = X_train[split_idx:]
            y_val = y_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx]
        
        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=TRAINING_CONFIG['num_workers'],
            pin_memory=TRAINING_CONFIG['pin_memory']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )
        
        # Setup training
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(self.config['epochs']):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for features, labels in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            val_loss, val_accuracy = self._validate(val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{self.config['epochs']} - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_accuracy:.4f}")
        
        self.is_trained = True
        
        return {
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_val_accuracy': history['val_accuracy'][-1],
            'best_val_loss': best_val_loss,
            'history': history
        }
    
    def _validate(self, val_loader, criterion) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)
                
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                predicted = (outputs >= 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        val_loss /= len(val_loader)
        accuracy = correct / total
        
        return val_loss, accuracy
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability predictions
        
        Args:
            X: Features
            
        Returns:
            Probability predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X.values).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy().flatten()
        
        return predictions
    
    def save(self, save_dir: str):
        """Save model to disk"""
        os.makedirs(save_dir, exist_ok=True)
        
        model_path = os.path.join(save_dir, f'{self.market}_dnn.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': self.input_size,
            'config': self.config,
            'best_loss': self.best_loss,
            'feature_names': self.feature_names
        }, model_path)
    
    def load(self, save_dir: str):
        """Load model from disk"""
        model_path = os.path.join(save_dir, f'{self.market}_dnn.pth')
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_loss = checkpoint['best_loss']
        self.feature_names = checkpoint['feature_names']
        self.is_trained = True
