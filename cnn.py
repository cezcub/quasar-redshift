import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os

class QuasarCNN(nn.Module):
    def __init__(self, input_size, conv_channels=[32, 64, 128, 256, 512, 1024], fc_sizes=[2048, 1024, 512, 256, 128, 64], dropout_rates=[0.3, 0.3]):
        super(QuasarCNN, self).__init__()
        
        # For tabular data, we'll reshape into a sequence and use 1D convolutions
        # This allows the CNN to learn local patterns in the feature space
        
        # First, we'll add a linear layer to expand the feature space
        self.input_projection = nn.Linear(input_size, 64)
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        in_channels = 1
        for out_channels in conv_channels:
            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
            self.bn_layers.append(nn.BatchNorm1d(out_channels))
            self.dropout_layers.append(nn.Dropout(dropout_rates[0]))  # conv dropout
            in_channels = out_channels
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        self.fc_dropout_layers = nn.ModuleList()
        
        prev_size = conv_channels[-1] if conv_channels else 64
        for fc_size in fc_sizes:
            self.fc_layers.append(nn.Linear(prev_size, fc_size))
            self.fc_dropout_layers.append(nn.Dropout(dropout_rates[1]))  # fc dropout
            prev_size = fc_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, 1)
        
    def forward(self, x):
        # Project input to higher dimensional space
        x = F.relu(self.input_projection(x))
        
        # Reshape for 1D convolution: (batch_size, channels, sequence_length)
        x = x.unsqueeze(1)  # Add channel dimension
        
        # Convolutional layers with batch norm and dropout
        for conv, bn, dropout in zip(self.conv_layers, self.bn_layers, self.dropout_layers):
            x = F.relu(bn(conv(x)))
            x = dropout(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)  # Remove the last dimension
        
        # Fully connected layers
        for fc, dropout in zip(self.fc_layers, self.fc_dropout_layers):
            x = F.relu(fc(x))
            x = dropout(x)
        
        # Output layer (no activation for regression)
        x = self.output_layer(x)
        
        return x

def train_cnn_model(X_train, y_train, X_test, y_test, epochs=200, batch_size=64, learning_rate=0.001):
    """
    Train the CNN model on the provided data
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(device)
    
    # Create model
    input_size = X_train.shape[1]
    model = QuasarCNN(input_size).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5)
    
    # Training loop
    model.train()
    train_losses = []
    val_losses = []
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 30
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Calculate average training loss
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor).item()
            val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_cnn_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        model.train()
        
        # Print progress
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    # Load best model
    model.load_state_dict(torch.load('best_cnn_model.pth'))
    
    return model, train_losses, val_losses

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy().flatten()
    
    # Calculate MAE
    mae = mean_absolute_error(y_test, predictions)
    
    # Calculate other metrics
    bias = predictions - y_test
    normalized_bias = bias / (1 + y_test)
    scaled_mad_norm_bias = 1.4826 * np.median(np.abs(normalized_bias - np.median(normalized_bias)))
    
    return {
        'predictions': predictions,
        'mae': mae,
        'bias_mean': np.mean(bias),
        'bias_median': np.median(bias),
        'normalized_bias_mean': np.mean(normalized_bias),
        'normalized_bias_median': np.median(normalized_bias),
        'scaled_mad_normalized_bias': scaled_mad_norm_bias
    }