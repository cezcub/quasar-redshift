import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Conv Block adapted for 1D tabular data"""
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, se_ratio=0.25, dropout_rate=0.2):
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        self.dropout_rate = dropout_rate
        
        hidden_dim = in_channels * expand_ratio
        
        # Expansion phase
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU()
            )
        else:
            self.expand_conv = None
        
        # Depthwise convolution (adapted to 1D)
        self.depthwise_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride, 
                     padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU()
        )
        
        # Squeeze-and-Excitation
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(hidden_dim, se_channels, 1),
                nn.SiLU(),
                nn.Conv1d(se_channels, hidden_dim, 1),
                nn.Sigmoid()
            )
        else:
            self.se = None
        
        # Output projection
        self.project_conv = nn.Sequential(
            nn.Linear(hidden_dim, out_channels),
            nn.BatchNorm1d(out_channels)
        )
        
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        identity = x
        
        # Expansion
        if self.expand_conv is not None:
            x = self.expand_conv(x)
        
        # Reshape for 1D conv: (batch, features) -> (batch, features, 1)
        x = x.unsqueeze(-1)
        x = self.depthwise_conv(x)
        
        # Squeeze-and-Excitation
        if self.se is not None:
            se_weights = self.se(x)
            x = x * se_weights
        
        # Back to original shape: (batch, features, 1) -> (batch, features)
        x = x.squeeze(-1)
        x = self.project_conv(x)
        
        # Dropout
        if self.dropout_rate > 0 and self.training:
            x = self.dropout(x)
        
        # Residual connection
        if self.use_residual:
            x = x + identity
            
        return x


class EfficientNetTabular(nn.Module):
    """EfficientNet adapted for tabular regression (quasar flux -> redshift)"""
    
    def __init__(self, input_dim, num_classes=1, width_coefficient=1.2, depth_coefficient=1.4, 
                 dropout_rate=0.3, drop_connect_rate=0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.drop_connect_rate = drop_connect_rate
        
        # Calculate scaled dimensions
        def round_filters(filters):
            return int(filters * width_coefficient)
        
        def round_repeats(repeats):
            return int(math.ceil(repeats * depth_coefficient))
        
        # Stem - project input to higher dimension
        stem_size = round_filters(32)
        self.stem = nn.Sequential(
            nn.Linear(input_dim, stem_size),
            nn.BatchNorm1d(stem_size),
            nn.SiLU()
        )
        
        # MBConv blocks configuration (filters, repeats, stride, expand_ratio, kernel_size, se_ratio)
        block_configs = [
            (round_filters(16), round_repeats(1), 1, 1, 3, 0.25),
            (round_filters(24), round_repeats(2), 1, 6, 3, 0.25),
            (round_filters(40), round_repeats(2), 1, 6, 5, 0.25),
            (round_filters(80), round_repeats(3), 1, 6, 3, 0.25),
            (round_filters(112), round_repeats(3), 1, 6, 5, 0.25),
            (round_filters(192), round_repeats(4), 1, 6, 5, 0.25),
            (round_filters(320), round_repeats(1), 1, 6, 3, 0.25),
        ]
        
        # Build blocks
        self.blocks = nn.ModuleList()
        in_channels = stem_size
        
        for i, (out_channels, repeats, stride, expand_ratio, kernel_size, se_ratio) in enumerate(block_configs):
            # Calculate drop connect rate for this stage
            stage_drop_rate = self.drop_connect_rate * i / len(block_configs)
            
            for j in range(repeats):
                block_stride = stride if j == 0 else 1
                self.blocks.append(
                    MBConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        expand_ratio=expand_ratio,
                        kernel_size=kernel_size,
                        stride=block_stride,
                        se_ratio=se_ratio,
                        dropout_rate=stage_drop_rate
                    )
                )
                in_channels = out_channels
        
        # Head - more conservative sizing for tabular regression
        head_size = min(round_filters(512), 512)  # Cap the head size
        self.head = nn.Sequential(
            nn.Linear(in_channels, head_size),
            nn.BatchNorm1d(head_size),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(head_size, head_size // 4),
            nn.BatchNorm1d(head_size // 4),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(head_size // 4, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Xavier initialization for better gradient flow in regression
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.Conv1d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize final layer with small weights for regression
        final_layer = self.head[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
            if final_layer.bias is not None:
                nn.init.zeros_(final_layer.bias)
    
    def forward(self, x):
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.head(x)
        return x.squeeze(-1) if x.shape[-1] == 1 else x


def efficientnet_b3_tabular(input_dim, num_classes=1):
    """EfficientNet-B3 adapted for tabular data"""
    return EfficientNetTabular(
        input_dim=input_dim,
        num_classes=num_classes,
        width_coefficient=1.2,
        depth_coefficient=1.4,
        dropout_rate=0.3,
        drop_connect_rate=0.2
    )


def efficientnet_b4_tabular(input_dim, num_classes=1):
    """EfficientNet-B4 adapted for tabular data"""
    return EfficientNetTabular(
        input_dim=input_dim,
        num_classes=num_classes,
        width_coefficient=1.4,
        depth_coefficient=1.8,
        dropout_rate=0.4,
        drop_connect_rate=0.2
    )


class QuasarEfficientNet(EfficientNetTabular):
    """Specialized EfficientNet for quasar redshift prediction"""
    
    def __init__(self, input_dim, model_size='b3', **kwargs):
        if model_size == 'b3':
            width_coeff, depth_coeff = 1.2, 1.4
        elif model_size == 'b4':
            width_coeff, depth_coeff = 1.4, 1.8
        else:
            raise ValueError("model_size must be 'b3' or 'b4'")
        
        super().__init__(
            input_dim=input_dim,
            num_classes=1,
            width_coefficient=width_coeff,
            depth_coefficient=depth_coeff,
            **kwargs
        )
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    input_dim = 15  # Number of flux features
    batch_size = 32
    
    # Test EfficientNet-B3
    model_b3 = QuasarEfficientNet(input_dim=input_dim, model_size='b3')
    print(f"EfficientNet-B3 parameters: {model_b3.count_parameters():,}")
    
    # Test EfficientNet-B4
    model_b4 = QuasarEfficientNet(input_dim=input_dim, model_size='b4')
    print(f"EfficientNet-B4 parameters: {model_b4.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(batch_size, input_dim)
    
    with torch.no_grad():
        output_b3 = model_b3(x)
        output_b4 = model_b4(x)
    
    print(f"Input shape: {x.shape}")
    print(f"EfficientNet-B3 output shape: {output_b3.shape}")
    print(f"EfficientNet-B4 output shape: {output_b4.shape}")
    print("✅ EfficientNet models working correctly!")
