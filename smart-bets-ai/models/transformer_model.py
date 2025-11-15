"""
Transformer Neural Network
Advanced attention mechanisms for complex relationship modeling
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
import math

from .config import TRANSFORMER_CONFIG, TRAINING_CONFIG


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerNetwork(nn.Module):
    """Transformer architecture for match prediction"""
    
    def __init__(self, input_size: int, d_model: int, nhead: int, 
                 num_encoder_layers: int, dim_feedforward: int, dropout: float):
        super(TransformerNetwork, self).__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Output layers
        self.fc1 = nn.Linear(d_model, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Output layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        
        return x


class TransformerPredictor:
    """
    Transformer-based predictor for match data
    Uses attention mechanisms to model complex relationships
    """
    
    def __init__(self, market: str, input_size: int):
        """
        Initialize Transformer predictor
        
        Args:
            market: Target market ('goals', 'cards', 'corners', 'btts')
            input_size: Number of features per timestep
        """
        self.market = market
        self.input_size = input_size
        self.config = TRANSFORMER_CONFIG
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and TRAINING_CONFIG['use_gpu'] else 'cpu')
        
        self.model = TransformerNetwork(
            input_size=input_size,
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            num_encoder_layers=self.config['num_encoder_layers'],
            dim_feedforward=self.config['dim_feedforward'],
            dropout=self.config['dropout']
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
        Train Transformer model
        
        Args:
            X: Training features
            y: Training labels
            validation_data: Optional (X_val, y_val) tuple
            
        Returns:
            Training metrics
        """
        from .lstm_model import MatchSequenceDataset
        
        # Prepare sequences
        X_seq, y_seq = self.prepare_sequences(X, y)
        
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_seq, y_val_seq = self.prepare_sequences(X_val, y_val)
        else:
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
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
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
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
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
        
        model_path = os.path.join(save_dir, f'{self.market}_transformer.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': self.input_size,
            'config': self.config,
            'best_loss': self.best_loss
        }, model_path)
    
    def load(self, save_dir: str):
        """Load model from disk"""
        model_path = os.path.join(save_dir, f'{self.market}_transformer.pth')
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_loss = checkpoint['best_loss']
        self.is_trained = True
