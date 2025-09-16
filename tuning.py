from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import optuna
import json
import time
from datetime import datetime

# Import all our models
from predict import load_data
from transformer import QuasarTransformer, create_data_loaders
from efficientnet import QuasarEfficientNet
from convnext import QuasarConvNeXt, ConvNeXtWithAttention
from cnn_transformer_hybrid import QuasarCNNTransformer, AdaptiveCNNTransformer
from vit import QuasarViT

# Load and prepare data
print("Loading quasar data...")
data = load_data('Sep<20.csv')

features = ['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z', 'FLUX_IVAR_W1', 'FLUX_W1', 'FLUX_W2', 'ML_FLUX_P1', 'ML_FLUX_P2', 'ML_FLUX_P3', 'ML_FLUX_P4', 'ML_FLUX_P5', 'ML_FLUX_P6']

X = data[features].values
Y = data['Z'].values

# Apply log transformation to all features
X = np.log(X + 1e-10)

# Remove any rows with NaN or infinite values after log transformation
mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1) | np.isnan(Y))
X = X[mask]
Y = Y[mask]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize features for neural networks
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Input features: {X_train.shape[1]}")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==================== TRAINING FUNCTIONS ====================

def train_model_with_params(model, X_train, y_train, X_val, y_val, train_params, epochs=50):
    """Generic training function for all model types"""
    model = model.to(device)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # Training setup
    criterion = nn.MSELoss()
    
    if 'optimizer_type' in train_params and train_params['optimizer_type'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=train_params['learning_rate'],
            weight_decay=train_params['weight_decay'],
            momentum=train_params.get('momentum', 0.9)
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_params['learning_rate'],
            weight_decay=train_params['weight_decay']
        )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, verbose=False
    )
    
    # Create data loaders for batch training
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_params['batch_size'], shuffle=True
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            if 'grad_clip_norm' in train_params:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_params['grad_clip_norm'])
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    return best_val_loss

# ==================== ARCHITECTURE-SPECIFIC OBJECTIVES ====================

def efficientnet_objective(trial):
    """Optuna objective for EfficientNet"""
    model_params = {
        'model_size': trial.suggest_categorical('model_size', ['b3', 'b4']),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'drop_connect_rate': trial.suggest_float('drop_connect_rate', 0.1, 0.3)
    }
    
    train_params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-2, log=True),
        'grad_clip_norm': trial.suggest_float('grad_clip_norm', 0.5, 2.0),
        'optimizer_type': trial.suggest_categorical('optimizer_type', ['adamw', 'sgd'])
    }
    
    if train_params['optimizer_type'] == 'sgd':
        train_params['momentum'] = trial.suggest_float('momentum', 0.8, 0.99)
    
    try:
        model = QuasarEfficientNet(input_dim=X_train_scaled.shape[1], **model_params)
        val_loss = train_model_with_params(model, X_train_scaled, y_train, X_test_scaled, y_test, train_params, epochs=40)
        return val_loss
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')

def convnext_objective(trial):
    """Optuna objective for ConvNeXt"""
    model_params = {
        'model_size': trial.suggest_categorical('model_size', ['tiny', 'small', 'base']),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.4),
        'drop_path_rate': trial.suggest_float('drop_path_rate', 0.0, 0.3),
        'layer_scale_init_value': trial.suggest_float('layer_scale_init_value', 1e-7, 1e-5, log=True)
    }
    
    train_params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-2, log=True),
        'grad_clip_norm': trial.suggest_float('grad_clip_norm', 0.5, 2.0),
        'optimizer_type': trial.suggest_categorical('optimizer_type', ['adamw', 'sgd'])
    }
    
    if train_params['optimizer_type'] == 'sgd':
        train_params['momentum'] = trial.suggest_float('momentum', 0.8, 0.99)
    
    use_attention = trial.suggest_categorical('use_attention', [True, False])
    
    try:
        if use_attention:
            model = ConvNeXtWithAttention(input_dim=X_train_scaled.shape[1], **model_params)
        else:
            model = QuasarConvNeXt(input_dim=X_train_scaled.shape[1], **model_params)
        
        val_loss = train_model_with_params(model, X_train_scaled, y_train, X_test_scaled, y_test, train_params, epochs=40)
        return val_loss
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')

def cnn_transformer_objective(trial):
    """Optuna objective for CNN-Transformer Hybrid"""
    model_params = {
        'model_size': trial.suggest_categorical('model_size', ['small', 'base', 'large']),
        'fusion_method': trial.suggest_categorical('fusion_method', ['attention', 'concatenation', 'addition']),
        'cnn_layers': trial.suggest_int('cnn_layers', 2, 5),
        'dropout': trial.suggest_float('dropout', 0.1, 0.4)
    }
    
    train_params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-2, log=True),
        'grad_clip_norm': trial.suggest_float('grad_clip_norm', 0.5, 2.0),
        'optimizer_type': trial.suggest_categorical('optimizer_type', ['adamw', 'sgd'])
    }
    
    if train_params['optimizer_type'] == 'sgd':
        train_params['momentum'] = trial.suggest_float('momentum', 0.8, 0.99)
    
    use_adaptive = trial.suggest_categorical('use_adaptive', [True, False])
    
    try:
        if use_adaptive:
            model = AdaptiveCNNTransformer(input_dim=X_train_scaled.shape[1], **model_params)
        else:
            model = QuasarCNNTransformer(input_dim=X_train_scaled.shape[1], **model_params)
        
        val_loss = train_model_with_params(model, X_train_scaled, y_train, X_test_scaled, y_test, train_params, epochs=40)
        return val_loss
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')

def vit_objective(trial):
    """Optuna objective for Vision Transformer"""
    model_params = {
        'model_size': trial.suggest_categorical('model_size', ['tiny', 'small', 'base']),
        'patch_size': trial.suggest_int('patch_size', 1, max(1, X_train_scaled.shape[1] // 4)),
        'dropout': trial.suggest_float('dropout', 0.1, 0.4),
        'attention_dropout': trial.suggest_float('attention_dropout', 0.0, 0.3),
        'drop_path_rate': trial.suggest_float('drop_path_rate', 0.0, 0.3),
        'mlp_ratio': trial.suggest_int('mlp_ratio', 2, 6),
        'use_cls_token': trial.suggest_categorical('use_cls_token', [True, False])
    }
    
    train_params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-2, log=True),
        'grad_clip_norm': trial.suggest_float('grad_clip_norm', 0.5, 2.0),
        'optimizer_type': trial.suggest_categorical('optimizer_type', ['adamw', 'sgd'])
    }
    
    if train_params['optimizer_type'] == 'sgd':
        train_params['momentum'] = trial.suggest_float('momentum', 0.8, 0.99)
    
    try:
        model = QuasarViT(input_dim=X_train_scaled.shape[1], **model_params)
        val_loss = train_model_with_params(model, X_train_scaled, y_train, X_test_scaled, y_test, train_params, epochs=40)
        return val_loss
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')

# ==================== MAIN OPTIMIZATION LOOP ====================

def run_architecture_optimization(architecture_name, objective_func, n_trials=200):
    """Run optimization for a specific architecture"""
    print(f"\n{'='*60}")
    print(f"Starting optimization for {architecture_name}")
    print(f"{'='*60}")
    
    study = optuna.create_study(direction='minimize')
    
    start_time = time.time()
    study.optimize(objective_func, n_trials=n_trials)
    end_time = time.time()
    
    print(f"\n{architecture_name} optimization completed!")
    print(f"Time taken: {(end_time - start_time) / 60:.2f} minutes")
    print(f"Best validation loss: {study.best_value:.6f}")
    print(f"Best parameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    
    # Save results
    results_data = []
    for i, trial in enumerate(study.trials):
        result_dict = {
            'trial': i,
            'value': trial.value,
            'state': str(trial.state),
            **trial.params
        }
        results_data.append(result_dict)
    
    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values('value')
    
    # Save to CSV
    filename = f'{architecture_name.lower().replace(" ", "_").replace("-", "_")}_optuna_results.csv'
    results_df.to_csv(filename, index=False)
    print(f"Detailed results saved to '{filename}'")
    
    return study

# ==================== RUN ALL OPTIMIZATIONS ====================

if __name__ == "__main__":
    
    architectures = [
        ("EfficientNet", efficientnet_objective),
        ("ConvNeXt", convnext_objective),
        ("CNN_Transformer", cnn_transformer_objective),
        ("Vision_Transformer", vit_objective)
    ]
    
    all_results = {}
    
    for arch_name, objective_func in architectures:
        try:
            study = run_architecture_optimization(arch_name, objective_func, n_trials=200)
            all_results[arch_name] = {
                'best_value': study.best_value,
                'best_params': study.best_params,
                'n_trials': len(study.trials)
            }
        except Exception as e:
            print(f"Error optimizing {arch_name}: {e}")
            all_results[arch_name] = {
                'best_value': float('inf'),
                'best_params': {},
                'n_trials': 0,
                'error': str(e)
            }
    
    # ==================== FINAL COMPARISON ====================
    
    print(f"\n{'='*80}")
    print("FINAL ARCHITECTURE COMPARISON")
    print(f"{'='*80}")
    
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['best_value'])
    
    for i, (arch_name, results) in enumerate(sorted_results, 1):
        if 'error' not in results:
            print(f"{i}. {arch_name}: {results['best_value']:.6f} (Best validation loss)")
        else:
            print(f"{i}. {arch_name}: FAILED - {results['error']}")
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_filename = f'architecture_comparison_{timestamp}.json'
    
    with open(summary_filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nComprehensive results saved to '{summary_filename}'")
    print(f"\n🎉 Multi-architecture optimization complete!")
    print(f"Best overall architecture: {sorted_results[0][0]} with validation loss: {sorted_results[0][1]['best_value']:.6f}")
