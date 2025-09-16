import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding1D(nn.Module):
    """1D Positional encoding for tabular features"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class CNNFeatureExtractor(nn.Module):
    """CNN component for local feature extraction"""
    
    def __init__(self, input_dim, output_dim=256, num_layers=3):
        super().__init__()
        
        # Calculate intermediate dimension based on output_dim
        intermediate_dim = min(64, output_dim // 2)
        self.input_projection = nn.Linear(input_dim, intermediate_dim)
        
        # 1D CNN layers for local pattern extraction
        cnn_layers = []
        in_channels = 1
        # Adjust channel progression to reach output_dim
        if num_layers == 3:
            channels = [32, 64, output_dim]
        elif num_layers == 4:
            channels = [32, 64, 128, output_dim]
        else:
            # Default progression for other cases
            step = max(32, output_dim // (num_layers + 1))
            channels = [min(32 + i * step, output_dim) for i in range(num_layers)]
            channels[-1] = output_dim
        
        for i in range(num_layers):
            out_channels = channels[i]
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Adaptive pooling to ensure consistent output size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(input_dim)
        
        # Final projection - input size should match the CNN output
        self.output_projection = nn.Linear(output_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch, features)
        batch_size, features = x.shape
        
        # Project and reshape for CNN
        x = F.relu(self.input_projection(x))  # (batch, intermediate_dim)
        x = x.unsqueeze(1)  # (batch, 1, intermediate_dim)
        
        # Apply CNN
        x = self.cnn(x)  # (batch, output_dim, intermediate_dim)
        
        # Adaptive pooling to match original feature count
        x = self.adaptive_pool(x)  # (batch, output_dim, features)
        
        # Transpose for transformer: (batch, features, output_dim)
        x = x.transpose(1, 2)
        
        # Final projection
        x = self.output_projection(x)
        
        return x


class TransformerEncoder(nn.Module):
    """Transformer component for global attention"""
    
    def __init__(self, d_model, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding1D(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)
        
        # Apply transformer
        x = self.transformer(x)
        x = self.norm(x)
        
        return x


class CNNTransformerHybrid(nn.Module):
    """Hybrid CNN-Transformer for tabular regression"""
    
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=4, 
                 cnn_layers=3, dim_feedforward=512, dropout=0.1, 
                 fusion_method='attention'):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.fusion_method = fusion_method
        
        # CNN feature extractor
        self.cnn_extractor = CNNFeatureExtractor(
            input_dim=input_dim,
            output_dim=d_model,
            num_layers=cnn_layers
        )
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Fusion mechanism
        if fusion_method == 'attention':
            self.fusion = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True
            )
        elif fusion_method == 'concatenation':
            self.fusion = nn.Linear(d_model * 2, d_model)
        elif fusion_method == 'addition':
            self.fusion = None  # Simple addition
        else:
            raise ValueError("fusion_method must be 'attention', 'concatenation', or 'addition'")
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Regression head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 4, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        # CNN feature extraction
        cnn_features = self.cnn_extractor(x)  # (batch, seq_len, d_model)
        
        # Transformer processing
        transformer_features = self.transformer(cnn_features)  # (batch, seq_len, d_model)
        
        # Fusion
        if self.fusion_method == 'attention':
            # Use transformer features as query, CNN features as key/value
            fused_features, _ = self.fusion(
                transformer_features, cnn_features, cnn_features
            )
        elif self.fusion_method == 'concatenation':
            # Concatenate along feature dimension
            combined = torch.cat([cnn_features, transformer_features], dim=-1)
            fused_features = self.fusion(combined)
        else:  # addition
            fused_features = cnn_features + transformer_features
        
        # Global pooling: (batch, seq_len, d_model) -> (batch, d_model)
        fused_features = fused_features.transpose(1, 2)  # (batch, d_model, seq_len)
        pooled = self.global_pool(fused_features).squeeze(-1)  # (batch, d_model)
        
        # Regression output
        output = self.head(pooled)
        return output.squeeze(-1) if output.shape[-1] == 1 else output


class QuasarCNNTransformer(CNNTransformerHybrid):
    """Specialized CNN-Transformer hybrid for quasar redshift prediction"""
    
    def __init__(self, input_dim, model_size='base', fusion_method='attention', **kwargs):
        
        if model_size == 'small':
            d_model, nhead, num_layers = 128, 4, 2
        elif model_size == 'base':
            d_model, nhead, num_layers = 256, 8, 4
        elif model_size == 'large':
            d_model, nhead, num_layers = 384, 12, 6
        else:
            raise ValueError("model_size must be 'small', 'base', or 'large'")
        
        super().__init__(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            fusion_method=fusion_method,
            **kwargs
        )
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_attention_weights(self, x):
        """Extract attention weights for interpretability"""
        attention_weights = []
        
        def hook_fn(module, input, output):
            if len(output) > 1:
                attention_weights.append(output[1])
        
        # Register hooks for transformer layers
        hooks = []
        for layer in self.transformer.transformer.layers:
            hook = layer.self_attn.register_forward_hook(hook_fn)
            hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.forward(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_weights


class AdaptiveCNNTransformer(QuasarCNNTransformer):
    """CNN-Transformer with adaptive feature weighting"""
    
    def __init__(self, input_dim, **kwargs):
        super().__init__(input_dim, **kwargs)
        
        # Add feature importance learning
        self.feature_gate = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )
        
        # Add uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 4),
            nn.ReLU(),
            nn.Linear(self.d_model // 4, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
    
    def forward(self, x, return_uncertainty=False):
        # Learn adaptive feature weights
        feature_weights = self.feature_gate(x)
        weighted_x = x * feature_weights
        
        # Standard CNN-Transformer forward pass
        cnn_features = self.cnn_extractor(weighted_x)
        transformer_features = self.transformer(cnn_features)
        
        # Fusion
        if self.fusion_method == 'attention':
            fused_features, attn_weights = self.fusion(
                transformer_features, cnn_features, cnn_features
            )
        else:
            fused_features = cnn_features + transformer_features
            attn_weights = None
        
        # Global pooling
        fused_features = fused_features.transpose(1, 2)
        pooled = self.global_pool(fused_features).squeeze(-1)
        
        # Predictions
        prediction = self.head(pooled).squeeze(-1)
        
        if return_uncertainty:
            uncertainty = self.uncertainty_head(pooled).squeeze(-1)
            return prediction, uncertainty, feature_weights
        
        return prediction


if __name__ == "__main__":
    # Test the models
    input_dim = 15  # Number of flux features
    batch_size = 32
    
    # Test base CNN-Transformer
    model_base = QuasarCNNTransformer(input_dim=input_dim, model_size='base')
    print(f"CNN-Transformer Base parameters: {model_base.count_parameters():,}")
    
    # Test adaptive version
    model_adaptive = AdaptiveCNNTransformer(input_dim=input_dim, model_size='base')
    print(f"Adaptive CNN-Transformer parameters: {model_adaptive.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(batch_size, input_dim)
    
    with torch.no_grad():
        output_base = model_base(x)
        output_adaptive, uncertainty, feature_weights = model_adaptive(x, return_uncertainty=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Base output shape: {output_base.shape}")
    print(f"Adaptive output shape: {output_adaptive.shape}")
    print(f"Uncertainty shape: {uncertainty.shape}")
    print(f"Feature weights shape: {feature_weights.shape}")
    print("✅ CNN-Transformer hybrid models working correctly!")
