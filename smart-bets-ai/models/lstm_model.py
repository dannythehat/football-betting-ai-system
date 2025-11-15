"""
LSTM Neural Network
Time-series pattern recognition for sequential match data
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import pickle
import os

from .config import LSTM_CONFIG, TRAINING_CONFIG


class MatchSequenceDataset(Dataset):
    """Dataset for sequential match data"""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class LSTMNetwork(nn.Module):
    """LSTM architecture for match prediction"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 dropout: float, bidirectional: bool):
        super(LSTMNetwork, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Fully connected layers
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc1 = nn.Linear(lstm_output_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.sigmoid(self.fc3(out))
        
        return out


class LSTMPredictor:
    """
    LSTM-based predictor for time-series match data
    Captures sequential patterns and momentum
    """
    
    def __init__(self, market: str, input_size: int):
        """
        Initialize LSTM predictor
        
        Args:
            market: Target market ('goals', 'cards', 'corners', 'btts')
            input_size: Number of features per timestep
        """
        self.market = market
        self.input_size = input_size
        self.config = LSTM_CONFIG
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and TRAINING_CONFIG['use_gpu'] else 'cpu')
        
        self.model = LSTMNetwork(
            input_size=input_size,
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout'],
            bidirectional=self.config['bidirectional']
        ).to(self.device)
        
        self.is_trained = False
        self.best_loss = float('inf')
    
    def prepare_sequences(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple:
        """
        Convert tabular data to sequences
        
        Args:
            X: Features (assumes data is sorted by time)
            y: Labels (optional)
            
        Returns:
            Sequences and labels (if provided)
        """
        sequence_length = self.config['sequence_length']
        
        # For simplicity, we'll create sequences by sliding window
        # In production, you'd group by team/match and create proper sequences
        sequences = []
        labels = []
        
        for i in range(len(X) - sequence_length + 1):
            seq = X.iloc[i:i+sequence_length].values
            sequences.append(seq)
            
            if y is not None:
                labels.append(y.iloc[i+sequence_length-1])
        
        sequences = np.array(sequences)
        labels = np.array(labels) if y is not None else None
        
        return sequences, labels
    
    def train(self, X: pd.DataFrame, y: pd.Series,
              validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None) -> Dict[str, Any]:
        """
        Train LSTM model
        
        Args:
            X: Training features
            y: Training labels
            validation_data: Optional (X_val, y_val) tuple
            
        Returns:
            Training metrics
        """
        # Prepare sequences
        X_seq, y_seq = self.prepare_sequences(X, y)
        
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_seq, y_val_seq = self.prepare_sequences(X_val, y_val)
        else:
            # Split validation
            split_idx = int(len(X_seq) * 0.8)
            X_val_seq = X_seq[split_idx:]
            y_val_seq = y_seq[split_idx:]
            X_seq = X_seq[:split_idx]
            y_seq = y_seq[:split_idx]
        
        # Create datasets
        train_dataset = MatchSequenceDataset(X_seq, y_seq)
        val_dataset = MatchSequenceDataset(X_val_seq, y_val_seq)
        
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
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(self.config['epochs']):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for sequences, labels in train_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            val_loss, val_accuracy = self._validate(val_loader, criterion)
            
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
            
            if (epoch + 1) % 10 == 0:
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
            for sequences, labels in val_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)
                
                outputs = self.model(sequences)
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
        
        X_seq, _ = self.prepare_sequences(X)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X_seq), self.config['batch_size']):
                batch = X_seq[i:i+self.config['batch_size']]
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                outputs = self.model(batch_tensor)
                predictions.extend(outputs.cpu().numpy().flatten())
        
        return np.array(predictions)
    
    def save(self, save_dir: str):
        """Save model to disk"""
        os.makedirs(save_dir, exist_ok=True)
        
        model_path = os.path.join(save_dir, f'{self.market}_lstm.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': self.input_size,
            'config': self.config,
            'best_loss': self.best_loss
        }, model_path)
    
    def load(self, save_dir: str):
        """Load model from disk"""
        model_path = os.path.join(save_dir, f'{self.market}_lstm.pth')
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_loss = checkpoint['best_loss']
        self.is_trained = True
