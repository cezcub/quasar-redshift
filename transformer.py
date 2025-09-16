import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer inputs.
    Since we're dealing with tabular data, this helps the model understand
    the importance/position of different features.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TabularTransformerRegressor(nn.Module):
    """
    Transformer model adapted for tabular regression tasks.
    This model treats each feature as a token in the sequence.
    """
    def __init__(self, 
                 input_dim,
                 d_model=128,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=512,
                 dropout=0.1,
                 activation='relu',
                 layer_norm_eps=1e-5):
        super(TabularTransformerRegressor, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Project input features to model dimension
        self.input_projection = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=input_dim, dropout=dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global pooling and output layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_layers = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        batch_size, seq_len = x.shape
        
        # Reshape to treat each feature as a token
        # (batch_size, seq_len, 1)
        x = x.unsqueeze(-1)
        
        # Project to model dimension
        # (batch_size, seq_len, d_model)
        x = self.input_projection(x)
        
        # Add positional encoding
        # Transpose for positional encoding (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        # Transpose back (batch_size, seq_len, d_model)
        x = x.transpose(0, 1)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Global pooling: (batch_size, d_model, seq_len) -> (batch_size, d_model, 1)
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x = self.global_pool(x)  # (batch_size, d_model, 1)
        x = x.squeeze(-1)  # (batch_size, d_model)
        
        # Output layers
        output = self.output_layers(x)
        
        return output.squeeze(-1)  # (batch_size,)


class QuasarTransformer(TabularTransformerRegressor):
    """
    Specialized transformer for quasar redshift prediction.
    This class provides convenient defaults and additional methods.
    All hyperparameters are customizable for tuning purposes.
    """
    def __init__(self, 
                 input_dim,
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 dim_feedforward=1024,
                 dropout=0.15,
                 output_layers=None,
                 activation='relu',
                 layer_norm_eps=1e-5,
                 **kwargs):
        
        # Store hyperparameters for easy access
        self.hyperparams = {
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
            'activation': activation,
            'layer_norm_eps': layer_norm_eps
        }
        
        # If custom output layers are provided, store them
        self.custom_output_layers = output_layers
        
        super().__init__(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            **kwargs
        )
        
        # Override output layers if custom ones are provided
        if output_layers is not None:
            self.output_layers = self._create_custom_output_layers(output_layers, d_model, dropout)
    
    def _create_custom_output_layers(self, layer_sizes, d_model, dropout):
        """Create custom output layers based on provided sizes"""
        layers = []
        
        # First layer from d_model to first hidden size
        if len(layer_sizes) > 0:
            layers.append(nn.Linear(d_model, layer_sizes[0]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
            # Hidden layers
            for i in range(len(layer_sizes) - 1):
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            
            # Final output layer
            layers.append(nn.Linear(layer_sizes[-1], 1))
        else:
            # Direct connection if no hidden layers specified
            layers.append(nn.Linear(d_model, 1))
        
        return nn.Sequential(*layers)
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_hyperparams(self):
        """Get current hyperparameters"""
        return self.hyperparams.copy()
    
    def get_attention_weights(self, x):
        """
        Extract attention weights from the model.
        Useful for interpretability.
        """
        attention_weights = []
        
        def hook_fn(module, input, output):
            # output[1] contains attention weights for TransformerEncoderLayer
            if len(output) > 1:
                attention_weights.append(output[1])
        
        hooks = []
        for layer in self.transformer_encoder.layers:
            hook = layer.self_attn.register_forward_hook(hook_fn)
            hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.forward(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_weights


def train_transformer(model, train_loader, val_loader, 
                     num_epochs=100, 
                     lr=0.001, 
                     weight_decay=1e-4,
                     scheduler_factor=0.5,
                     scheduler_patience=10,
                     early_stopping_patience=20,
                     grad_clip_norm=1.0,
                     device='cpu',
                     save_model_path='best_transformer_model.pth',
                     verbose=True):
    """
    Training function for the transformer model with customizable hyperparameters.
    
    Args:
        model: The transformer model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Maximum number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for AdamW optimizer
        scheduler_factor: Factor by which learning rate is reduced
        scheduler_patience: Number of epochs to wait before reducing LR
        early_stopping_patience: Number of epochs to wait before early stopping
        grad_clip_norm: Maximum norm for gradient clipping
        device: Device to train on ('cpu' or 'cuda')
        save_model_path: Path to save the best model
        verbose: Whether to print training progress
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=verbose
    )
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), save_model_path)
        else:
            patience_counter += 1
        
        if verbose and epoch % 10 == 0:
            print(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        if patience_counter >= early_stopping_patience:
            if verbose:
                print(f'Early stopping at epoch {epoch}')
            break
    
    # Load best model
    model.load_state_dict(torch.load(save_model_path))
    
    return train_losses, val_losses


def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=64):
    """
    Create PyTorch data loaders for training and validation.
    """
    from torch.utils.data import TensorDataset, DataLoader
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def create_transformer_config(config_name='default', **overrides):
    """
    Factory function to create transformer configurations for hyperparameter tuning.
    
    Args:
        config_name: Pre-defined configuration name
        **overrides: Any parameters to override in the configuration
    
    Returns:
        Dictionary of hyperparameters
    """
    configs = {
        'default': {
            'd_model': 256,
            'nhead': 8,
            'num_layers': 4,
            'dim_feedforward': 1024,
            'dropout': 0.15,
            'output_layers': [512, 256, 128],
            'activation': 'relu',
            'layer_norm_eps': 1e-5
        },
        'small': {
            'd_model': 128,
            'nhead': 4,
            'num_layers': 2,
            'dim_feedforward': 256,
            'dropout': 0.1,
            'output_layers': [256, 128],
            'activation': 'relu',
            'layer_norm_eps': 1e-5
        },
        'large': {
            'd_model': 512,
            'nhead': 16,
            'num_layers': 8,
            'dim_feedforward': 2048,
            'dropout': 0.2,
            'output_layers': [1024, 512, 256, 128],
            'activation': 'gelu',
            'layer_norm_eps': 1e-5
        },
        'deep': {
            'd_model': 256,
            'nhead': 8,
            'num_layers': 12,
            'dim_feedforward': 1024,
            'dropout': 0.25,
            'output_layers': [512, 256, 128, 64],
            'activation': 'gelu',
            'layer_norm_eps': 1e-5
        }
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config name: {config_name}. Available: {list(configs.keys())}")
    
    config = configs[config_name].copy()
    config.update(overrides)
    
    return config


def create_training_config(config_name='default', **overrides):
    """
    Factory function to create training configurations.
    
    Args:
        config_name: Pre-defined configuration name
        **overrides: Any parameters to override in the configuration
    
    Returns:
        Dictionary of training hyperparameters
    """
    configs = {
        'default': {
            'num_epochs': 100,
            'lr': 0.001,
            'weight_decay': 1e-4,
            'scheduler_factor': 0.5,
            'scheduler_patience': 10,
            'early_stopping_patience': 20,
            'grad_clip_norm': 1.0,
            'batch_size': 64
        },
        'fast': {
            'num_epochs': 50,
            'lr': 0.01,
            'weight_decay': 1e-3,
            'scheduler_factor': 0.3,
            'scheduler_patience': 5,
            'early_stopping_patience': 10,
            'grad_clip_norm': 0.5,
            'batch_size': 128
        },
        'careful': {
            'num_epochs': 200,
            'lr': 0.0005,
            'weight_decay': 1e-5,
            'scheduler_factor': 0.7,
            'scheduler_patience': 15,
            'early_stopping_patience': 30,
            'grad_clip_norm': 2.0,
            'batch_size': 32
        }
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config name: {config_name}. Available: {list(configs.keys())}")
    
    config = configs[config_name].copy()
    config.update(overrides)
    
    return config


if __name__ == "__main__":
    # Example usage demonstrating customizable hyperparameters
    print("Transformer model for tabular regression with customizable hyperparameters!")
    
    # Example 1: Using default configuration
    input_dim = 20  # Example: 20 features
    model_default = QuasarTransformer(input_dim=input_dim)
    print(f"\nDefault model created with {model_default.count_parameters():,} parameters")
    print(f"Default hyperparameters: {model_default.get_hyperparams()}")
    
    # Example 2: Using pre-defined small configuration
    small_config = create_transformer_config('small')
    model_small = QuasarTransformer(input_dim=input_dim, **small_config)
    print(f"\nSmall model created with {model_small.count_parameters():,} parameters")
    print(f"Small hyperparameters: {model_small.get_hyperparams()}")
    
    # Example 3: Custom configuration with overrides
    custom_config = create_transformer_config('default', 
                                            d_model=128, 
                                            num_layers=6, 
                                            dropout=0.2,
                                            output_layers=[256, 128, 64])
    model_custom = QuasarTransformer(input_dim=input_dim, **custom_config)
    print(f"\nCustom model created with {model_custom.count_parameters():,} parameters")
    print(f"Custom hyperparameters: {model_custom.get_hyperparams()}")
    
    # Example 4: Training configuration examples
    print(f"\nAvailable training configurations:")
    for config_name in ['default', 'fast', 'careful']:
        train_config = create_training_config(config_name)
        print(f"{config_name}: lr={train_config['lr']}, epochs={train_config['num_epochs']}, batch_size={train_config['batch_size']}")
    
    # Test with random data
    batch_size = 32
    x_sample = torch.randn(batch_size, input_dim)
    output = model_default(x_sample)
    print(f"\nTesting:")
    print(f"Input shape: {x_sample.shape}")
    print(f"Output shape: {output.shape}")
    
    print(f"\nModel ready for hyperparameter tuning!")