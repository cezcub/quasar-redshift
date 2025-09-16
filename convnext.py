import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm1d(nn.Module):
    """1D LayerNorm for tabular data"""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block adapted for 1D tabular data"""
    
    def __init__(self, dim, layer_scale_init_value=1e-6, drop_path_rate=0.0):
        super().__init__()
        
        # Depthwise convolution (adapted to 1D)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        # LayerNorm
        self.norm = LayerNorm1d(dim)
        
        # Pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # Layer scale
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), 
            requires_grad=True
        ) if layer_scale_init_value > 0 else None
        
        # Drop path
        self.drop_path_rate = drop_path_rate

    def drop_path(self, x, training=False):
        """Drop paths (Stochastic Depth) per sample"""
        if self.drop_path_rate == 0. or not training:
            return x
        keep_prob = 1 - self.drop_path_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

    def forward(self, x):
        input_x = x
        
        # Reshape for 1D conv: (batch, features) -> (batch, features, 1)
        x = x.unsqueeze(-1)
        x = self.dwconv(x)
        x = x.squeeze(-1)  # Back to (batch, features)
        
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        if self.gamma is not None:
            x = self.gamma * x
        
        x = self.drop_path(x, self.training)
        x = input_x + x
        
        return x


class ConvNeXtTabular(nn.Module):
    """ConvNeXt adapted for tabular regression (quasar flux -> redshift)"""
    
    def __init__(self, input_dim, num_classes=1, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0.0, layer_scale_init_value=1e-6, head_init_scale=1.0):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Stem - project input to first stage dimension
        self.stem = nn.Sequential(
            nn.Linear(input_dim, dims[0]),
            LayerNorm1d(dims[0])
        )
        
        # 4 feature resolution stages
        self.stages = nn.ModuleList()
        
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        for i in range(4):
            # Downsampling layer (except for first stage)
            if i > 0:
                downsample = nn.Sequential(
                    LayerNorm1d(dims[i-1]),
                    nn.Linear(dims[i-1], dims[i])
                )
            else:
                downsample = nn.Identity()
            
            # Stage blocks
            stage = nn.Sequential(
                downsample,
                *[ConvNeXtBlock(
                    dim=dims[i],
                    drop_path_rate=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        
        # Head
        self.norm = LayerNorm1d(dims[-1])
        self.head = nn.Sequential(
            nn.Linear(dims[-1], dims[-1] // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(dims[-1] // 2, dims[-1] // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dims[-1] // 4, num_classes)
        )
        
        # Initialize head
        self.head[-1].weight.data.mul_(head_init_scale)
        self.head[-1].bias.data.mul_(head_init_scale)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)
        
        for stage in self.stages:
            x = stage(x)
        
        x = self.norm(x)
        x = self.head(x)
        
        return x.squeeze(-1) if x.shape[-1] == 1 else x


def convnext_tiny_tabular(input_dim, num_classes=1, **kwargs):
    """ConvNeXt-Tiny adapted for tabular data"""
    return ConvNeXtTabular(
        input_dim=input_dim,
        num_classes=num_classes,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        **kwargs
    )


def convnext_small_tabular(input_dim, num_classes=1, **kwargs):
    """ConvNeXt-Small adapted for tabular data"""
    return ConvNeXtTabular(
        input_dim=input_dim,
        num_classes=num_classes,
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
        **kwargs
    )


def convnext_base_tabular(input_dim, num_classes=1, **kwargs):
    """ConvNeXt-Base adapted for tabular data"""
    return ConvNeXtTabular(
        input_dim=input_dim,
        num_classes=num_classes,
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        **kwargs
    )


class QuasarConvNeXt(ConvNeXtTabular):
    """Specialized ConvNeXt for quasar redshift prediction"""
    
    def __init__(self, input_dim, model_size='tiny', dropout_rate=0.3, **kwargs):
        
        if model_size == 'tiny':
            depths, dims = [3, 3, 9, 3], [96, 192, 384, 768]
        elif model_size == 'small':
            depths, dims = [3, 3, 27, 3], [96, 192, 384, 768]
        elif model_size == 'base':
            depths, dims = [3, 3, 27, 3], [128, 256, 512, 1024]
        else:
            raise ValueError("model_size must be 'tiny', 'small', or 'base'")
        
        super().__init__(
            input_dim=input_dim,
            num_classes=1,
            depths=depths,
            dims=dims,
            **kwargs
        )
        
        # Override head with custom dropout
        self.head = nn.Sequential(
            nn.Linear(dims[-1], dims[-1] // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dims[-1] // 2, dims[-1] // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(dims[-1] // 4, 1)
        )
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_feature_maps(self, x):
        """Extract feature maps from each stage for analysis"""
        features = []
        x = self.stem(x)
        features.append(x.clone())
        
        for stage in self.stages:
            x = stage(x)
            features.append(x.clone())
        
        return features


class ConvNeXtWithAttention(QuasarConvNeXt):
    """ConvNeXt with added attention mechanism for enhanced performance"""
    
    def __init__(self, input_dim, model_size='tiny', **kwargs):
        super().__init__(input_dim, model_size, **kwargs)
        
        # Get final dimension
        if model_size == 'base':
            final_dim = 1024
        else:
            final_dim = 768
        
        # Add attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=final_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Update head
        self.head = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(final_dim // 2, final_dim // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(final_dim // 4, 1)
        )
    
    def forward(self, x):
        x = self.stem(x)
        
        for stage in self.stages:
            x = stage(x)
        
        x = self.norm(x)
        
        # Apply attention (treating features as sequence length 1)
        x_attn = x.unsqueeze(1)  # (batch, 1, features)
        x_attn, _ = self.attention(x_attn, x_attn, x_attn)
        x = x_attn.squeeze(1)  # Back to (batch, features)
        
        x = self.head(x)
        return x.squeeze(-1) if x.shape[-1] == 1 else x


if __name__ == "__main__":
    # Test the models
    input_dim = 15  # Number of flux features
    batch_size = 32
    
    # Test ConvNeXt-Tiny
    model_tiny = QuasarConvNeXt(input_dim=input_dim, model_size='tiny')
    print(f"ConvNeXt-Tiny parameters: {model_tiny.count_parameters():,}")
    
    # Test ConvNeXt with Attention
    model_attn = ConvNeXtWithAttention(input_dim=input_dim, model_size='tiny')
    print(f"ConvNeXt-Tiny + Attention parameters: {model_attn.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(batch_size, input_dim)
    
    with torch.no_grad():
        output_tiny = model_tiny(x)
        output_attn = model_attn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"ConvNeXt-Tiny output shape: {output_tiny.shape}")
    print(f"ConvNeXt + Attention output shape: {output_attn.shape}")
    print("✅ ConvNeXt models working correctly!")
