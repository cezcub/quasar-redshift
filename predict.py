from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from cnn import QuasarCNN
from transformer import QuasarTransformer, create_data_loaders
from efficientnet import QuasarEfficientNet
from convnext import QuasarConvNeXt, ConvNeXtWithAttention
from cnn_transformer_hybrid import QuasarCNNTransformer, AdaptiveCNNTransformer
from vit import QuasarViT

# Load data
def load_data(file):
    data_folder = 'data'

    file_path = os.path.join(data_folder, file)
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        print(f"Loaded {file}: {len(data)} rows")

    return data

if __name__ == "__main__":

    data = load_data('Sep<20.csv')

    # Use ALL feature sets from model.py
    feature_columns = [
        ['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z', 'FLUX_W1', 'FLUX_W2'], 
        ['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z', 'FLUX_W1', 'FLUX_W2', 'ML_FLUX_P1', 'ML_FLUX_P2', 'ML_FLUX_P3', 'ML_FLUX_P4', 'ML_FLUX_P5', 'ML_FLUX_P6'], 
        ['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z', 'ML_FLUX_P1', 'ML_FLUX_P2', 'ML_FLUX_P3', 'ML_FLUX_P4', 'ML_FLUX_P5', 'ML_FLUX_P6'],
        ['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z', 'FLUX_W1', 'FLUX_W2', 'ML_FLUX_P1', 'ML_FLUX_P2', 'ML_FLUX_P3', 'ML_FLUX_P4', 'ML_FLUX_P5', 'ML_FLUX_P6'],
        ['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z', 'FLUX_W1', 'FLUX_W2', 'ML_FLUX_P1', 'ML_FLUX_P2', 'ML_FLUX_P3', 'ML_FLUX_P4', 'ML_FLUX_P5', 'ML_FLUX_P6'],
        ['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z', 'FLUX_W1', 'FLUX_W2', 'ML_FLUX_P1', 'ML_FLUX_P2', 'ML_FLUX_P3', 'ML_FLUX_P4', 'ML_FLUX_P5', 'ML_FLUX_P6'],
        ['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z', 'FLUX_W1', 'FLUX_W2', 'ML_FLUX_P1', 'ML_FLUX_P2', 'ML_FLUX_P3', 'ML_FLUX_P4', 'ML_FLUX_P5', 'ML_FLUX_P6'],
        ['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z', 'FLUX_W1', 'FLUX_W2', 'ML_FLUX_P1', 'ML_FLUX_P2', 'ML_FLUX_P3', 'ML_FLUX_P4', 'ML_FLUX_P5', 'ML_FLUX_P6'],
        ['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z', 'FLUX_W1', 'FLUX_W2', 'ML_FLUX_P1', 'ML_FLUX_P2', 'ML_FLUX_P3', 'ML_FLUX_P4', 'ML_FLUX_P5', 'ML_FLUX_P6']
    ]

    feature_set_names = ['IR', 'X-RAY', "NoIR", "ML_log_diff_G", "ML_log_ratio_G", "ML_log_diff_R", "ML_log_ratio_R", "ML_log_diff_Z", "ML_log_ratio_Z"]

    target_column = 'Z'  # Redshift

    # Store results for all feature sets
    all_results = []
    all_predictions = {}

    for feature_idx, features in enumerate(feature_columns):
        feature_name = feature_set_names[feature_idx]
        
        X = data[features].values
        y = data[target_column].values

        # Apply different transformations based on the feature set (same as model.py)
        if feature_name == "ML_log_diff_G":
            # For ML_log_diff_G: log(ML_FLUX) - log(FLUX_G) for ML_FLUX columns, regular log for others
            log_flux_g = np.log(data['FLUX_G'].values + 1e-10)
            
            for i, feature in enumerate(features):
                if feature.startswith('ML_FLUX'):
                    X[:, i] = np.log(X[:, i] + 1e-10) - log_flux_g
                else:
                    X[:, i] = np.log(X[:, i] + 1e-10)
                    
        elif feature_name == "ML_log_ratio_G":
            log_flux_g = np.log(data['FLUX_G'].values + 1e-10)
            
            for i, feature in enumerate(features):
                if feature.startswith('ML_FLUX'):
                    X[:, i] = np.log(X[:, i] + 1e-10) / log_flux_g
                else:
                    X[:, i] = np.log(X[:, i] + 1e-10)
                    
        elif feature_name == "ML_log_diff_R":
            log_flux_r = np.log(data['FLUX_R'].values + 1e-10)
            
            for i, feature in enumerate(features):
                if feature.startswith('ML_FLUX'):
                    X[:, i] = np.log(X[:, i] + 1e-10) - log_flux_r
                else:
                    X[:, i] = np.log(X[:, i] + 1e-10)
                    
        elif feature_name == "ML_log_ratio_R":
            log_flux_r = np.log(data['FLUX_R'].values + 1e-10)
            
            for i, feature in enumerate(features):
                if feature.startswith('ML_FLUX'):
                    X[:, i] = np.log(X[:, i] + 1e-10) / log_flux_r
                else:
                    X[:, i] = np.log(X[:, i] + 1e-10)
                    
        elif feature_name == "ML_log_diff_Z":
            log_flux_z = np.log(data['FLUX_Z'].values + 1e-10)
            
            for i, feature in enumerate(features):
                if feature.startswith('ML_FLUX'):
                    X[:, i] = np.log(X[:, i] + 1e-10) - log_flux_z
                else:
                    X[:, i] = np.log(X[:, i] + 1e-10)
                    
        elif feature_name == "ML_log_ratio_Z":
            log_flux_z = np.log(data['FLUX_Z'].values + 1e-10)
            
            for i, feature in enumerate(features):
                if feature.startswith('ML_FLUX'):
                    X[:, i] = np.log(X[:, i] + 1e-10) / log_flux_z
                else:
                    X[:, i] = np.log(X[:, i] + 1e-10)
        else:
            # For all other feature sets, apply regular log transformation
            X = np.log(X + 1e-10)

        # Remove any rows with NaN or infinite values after transformation
        mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]

        # Split data: 80% train, 20% test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")

        # Create multiple models for comparison
        models = {
            'RandomForest': ensemble.RandomForestRegressor(
                max_depth=50, 
                min_samples_split=2, 
                min_samples_leaf=2, 
                n_estimators=500, 
                bootstrap=True
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                verbosity=0
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.1,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                verbose=-1
            ),
            'CatBoost': CatBoostRegressor(
                iterations=500,
                depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bylevel=0.8,
                verbose=False
            ),
            'CNN': QuasarCNN(
                input_size=len(features),  # Will be set dynamically
                conv_channels=[32, 128, 64],
                fc_sizes=[512, 128, 128],
                dropout_rates=[0.18, 0.26]  # [conv_dropout, fc_dropout]
            ),
            'Transformer': QuasarTransformer(
                input_dim=len(features),  # Will be set dynamically
                d_model=512,
                nhead=8,
                num_layers=4,
                dim_feedforward=1024,
                dropout=0.050,
                activation='gelu',
                output_layers=[512, 512, 64]
            ),
            'EfficientNet': QuasarEfficientNet(
                input_dim=len(features),  # Will be set dynamically
                model_size='b4',
                dropout_rate=0.332,
                drop_connect_rate=0.196
            ),
            'ConvNeXt': QuasarConvNeXt(
                input_dim=len(features),  # Will be set dynamically
                model_size='tiny',
                dropout_rate=0.152,
                drop_path_rate=0.232,
                layer_scale_init_value=1.68e-07
            ),
            'ConvNeXt_Attention': ConvNeXtWithAttention(
                input_dim=len(features),  # Will be set dynamically
                model_size='tiny',
                dropout_rate=0.152,
                drop_path_rate=0.232,
                layer_scale_init_value=1.68e-07
            ),
            'CNN_Transformer': QuasarCNNTransformer(
                input_dim=len(features),  # Will be set dynamically
                model_size='base',
                fusion_method='attention',
                cnn_layers=5,
                dropout=0.2
            ),
            'Adaptive_CNN_Transformer': AdaptiveCNNTransformer(
                input_dim=len(features),  # Will be set dynamically
                model_size='base',
                fusion_method='attention',
                cnn_layers=5,
                dropout=0.2
            ),
            'ViT': QuasarViT(
                input_dim=len(features),  # Will be set dynamically
                model_size='small',
                dropout=0.2,
                attention_dropout=0.1,
                drop_path_rate=0.1,
                mlp_ratio=4,
                use_cls_token=True
            )
        }

        # Train and evaluate each model
        model_results = {}
        for model_name, model in models.items():
            print(f"Fitting {model_name} model for {feature_name}...")
            
            # TabNet requires special handling for validation data
            if model_name == 'TabNet':
                # Split training data further for TabNet validation
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42
                )
                model.fit(
                    X_train_split, y_train_split,
                    eval_set=[(X_val_split, y_val_split)],
                    max_epochs=200,
                    patience=20,
                    batch_size=1024,
                    drop_last=False
                )
                y_pred = model.predict(X_test)
            elif model_name == 'CNN':
                # CNN requires special handling with PyTorch
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # Standardize data for neural network
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Recreate model with correct input size
                model = QuasarCNN(
                    input_size=X_train_scaled.shape[1],
                    conv_channels=[32, 128, 64],
                    fc_sizes=[512, 128, 128],
                    dropout_rates=[0.18, 0.26]
                ).to(device)
                
                # Convert to tensors
                X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
                y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
                X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
                
                # Training setup
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.0025, weight_decay=3e-06)
                
                # Training
                model.train()
                train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
                
                epochs = 200
                for epoch in range(epochs):
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                
                # Make predictions
                model.eval()
                with torch.no_grad():
                    y_pred_tensor = model(X_test_tensor)
                    y_pred = y_pred_tensor.cpu().numpy().flatten()
            elif model_name == 'Transformer':
                # Transformer requires special handling with PyTorch
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # Standardize data for neural network
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Create transformer model with correct input dimension
                model = QuasarTransformer(
                    input_dim=X_train_scaled.shape[1],
                    d_model=512,
                    nhead=8,
                    num_layers=4,
                    dim_feedforward=1024,
                    dropout=0.050,
                    activation='gelu',
                    output_layers=[512, 512, 64]
                ).to(device)
                
                # Create data loaders
                train_loader, val_loader = create_data_loaders(
                    X_train_scaled, y_train, X_test_scaled, y_test, 
                    batch_size=32
                )
                
                # Training setup
                criterion = nn.MSELoss()
                optimizer = torch.optim.AdamW(
                    model.parameters(), 
                    lr=2.76e-05, 
                    weight_decay=7.08e-06
                )
                
                # Training
                model.train()
                epochs = 100
                for epoch in range(epochs):
                    for batch_x, batch_y in train_loader:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        
                        optimizer.zero_grad()
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.584)
                        optimizer.step()
                
                # Make predictions
                model.eval()
                with torch.no_grad():
                    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
                    y_pred_tensor = model(X_test_tensor)
                    y_pred = y_pred_tensor.cpu().numpy().flatten()
            elif model_name in ['EfficientNet', 'ConvNeXt', 'ConvNeXt_Attention', 'CNN_Transformer', 'Adaptive_CNN_Transformer', 'ViT']:
                # All new neural network models require similar PyTorch handling
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # Standardize data for neural network
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Create model with correct input dimension and optimal hyperparameters
                if model_name == 'EfficientNet':
                    model = QuasarEfficientNet(
                        input_dim=X_train_scaled.shape[1],
                        model_size='b4',
                        dropout_rate=0.332,
                        drop_connect_rate=0.196
                    ).to(device)
                    lr = 0.00973
                    weight_decay = 0.00125
                    batch_size = 64
                    grad_clip_norm = 0.584
                elif model_name == 'ConvNeXt':
                    model = QuasarConvNeXt(
                        input_dim=X_train_scaled.shape[1],
                        model_size='tiny',
                        dropout_rate=0.152,
                        drop_path_rate=0.232,
                        layer_scale_init_value=1.68e-07
                    ).to(device)
                    lr = 0.000161
                    weight_decay = 0.00841
                    batch_size = 32
                    grad_clip_norm = 1.702
                elif model_name == 'ConvNeXt_Attention':
                    model = ConvNeXtWithAttention(
                        input_dim=X_train_scaled.shape[1],
                        model_size='tiny',
                        dropout_rate=0.152,
                        drop_path_rate=0.232,
                        layer_scale_init_value=1.68e-07
                    ).to(device)
                    lr = 0.000161
                    weight_decay = 0.00841
                    batch_size = 32
                    grad_clip_norm = 1.702
                elif model_name == 'CNN_Transformer':
                    model = QuasarCNNTransformer(
                        input_dim=X_train_scaled.shape[1],
                        model_size='base',
                        fusion_method='attention',
                        cnn_layers=3,
                        dropout=0.2
                    ).to(device)
                    lr = 0.001
                    weight_decay = 1e-4
                    batch_size = 64
                    grad_clip_norm = 1.0
                elif model_name == 'Adaptive_CNN_Transformer':
                    model = AdaptiveCNNTransformer(
                        input_dim=X_train_scaled.shape[1],
                        model_size='base',
                        fusion_method='attention',
                        cnn_layers=3,
                        dropout=0.2
                    ).to(device)
                    lr = 0.001
                    weight_decay = 1e-4
                    batch_size = 64
                    grad_clip_norm = 1.0
                elif model_name == 'ViT':
                    # Let QuasarViT determine optimal patch_size automatically
                    model = QuasarViT(
                        input_dim=X_train_scaled.shape[1],
                        model_size='tiny',
                        # patch_size will be determined automatically by QuasarViT
                        dropout=0.2,
                        attention_dropout=0.1,
                        drop_path_rate=0.1,
                        mlp_ratio=4,
                        use_cls_token=True
                    ).to(device)
                    lr = 0.0005
                    weight_decay = 1e-4
                    batch_size = 64
                    grad_clip_norm = 1.0
                
                # Convert to tensors
                X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
                y_train_tensor = torch.FloatTensor(y_train).to(device)
                X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
                
                # Training setup
                criterion = nn.MSELoss()
                optimizer = torch.optim.AdamW(
                    model.parameters(), 
                    lr=lr, 
                    weight_decay=weight_decay
                )
                
                # Create data loader
                train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                
                # Training
                model.train()
                epochs = 100  # Reduced for faster evaluation
                for epoch in range(epochs):
                    for batch_x, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                        optimizer.step()
                
                # Make predictions
                model.eval()
                with torch.no_grad():
                    y_pred_tensor = model(X_test_tensor)
                    y_pred = y_pred_tensor.cpu().numpy().flatten()
            else:
                # Standard fit for other models
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate test MAE
            mae = mean_absolute_error(y_test, y_pred)
            print(f"{model_name} Test MAE: {mae:.6f}")
            
            # Store results
            model_results[model_name] = {
                'predictions': y_pred,
                'mae': mae
            }

        # Store predictions for each model
        for model_name, result in model_results.items():
            y_pred = result['predictions']
            mae = result['mae']
            
            # Store in all_predictions for this model/feature combination
            key = f"{feature_name}_{model_name}"
            all_predictions[key] = {
                'actual': y_test,
                'predicted': y_pred,
                'mae': mae
            }

            # Create predictions CSV for this model/feature combination
            predictions_df = pd.DataFrame({
                'Actual_Redshift': y_test,
                'Predicted_Redshift': y_pred,
                'Error': np.abs(y_test - y_pred)
            })

            # Calculate distribution statistics for this model/feature combination
            print(f"\nCalculating distribution statistics for {feature_name} with {model_name}...")
            
            # Calculate the specific statistics requested:
            # 1. Bias (prediction minus actual)
            bias = y_pred - y_test
            
            # 2. Normalized bias (bias divided by (1 + actual))
            normalized_bias = bias / (1 + y_test)
            
            # 3. Scaled median absolute deviation of the normalized bias
            # First calculate median absolute deviation (MAD)
            median_normalized_bias = np.median(normalized_bias)
            mad_normalized_bias = np.median(np.abs(normalized_bias - median_normalized_bias))
            # Scale factor for MAD to approximate standard deviation (for normal distribution)
            scaled_mad_normalized_bias = mad_normalized_bias * 1.4826
            
            # Store results
            result = {
                'Model': model_name,
                'Feature_Set': feature_name,
                'MAE': mae,
                'Bias_Mean': np.mean(bias),
                'Bias_Median': np.median(bias),
                'Normalized_Bias_Mean': np.mean(normalized_bias),
                'Normalized_Bias_Median': np.median(normalized_bias),
                'Scaled_MAD_Normalized_Bias': scaled_mad_normalized_bias,
                'Num_Predictions': len(y_test)
            }
            
            all_results.append(result)
            
            print(f"Bias Mean: {result['Bias_Mean']:.6f}")
            print(f"Bias Median: {result['Bias_Median']:.6f}")
            print(f"Normalized Bias Mean: {result['Normalized_Bias_Mean']:.6f}")
            print(f"Normalized Bias Median: {result['Normalized_Bias_Median']:.6f}")
            print(f"Scaled MAD of Normalized Bias: {result['Scaled_MAD_Normalized_Bias']:.6f}")

    # Create comprehensive results DataFrame
    results_df = pd.DataFrame(all_results)

    # Save comprehensive results
    results_df.to_csv('comprehensive_multi_model_results.csv', index=False)
    print(f"Results saved to 'comprehensive_multi_model_results.csv'")

    # Display summary table
    print(f"\nSUMMARY OF ALL MODEL/FEATURE COMBINATIONS:")
    print(results_df.round(6))

    # Group results by model for easier comparison
    print(f"\n" + "="*80)
    print("MODEL COMPARISON BY FEATURE SET:")
    print("="*80)

    for feature_set in results_df['Feature_Set'].unique():
        print(f"\nFeature Set: {feature_set}")
        feature_results = results_df[results_df['Feature_Set'] == feature_set].sort_values('MAE')
        for _, row in feature_results.iterrows():
            print(f"  {row['Model']:12s} - MAE: {row['MAE']:.6f}, Scaled MAD: {row['Scaled_MAD_Normalized_Bias']:.6f}")

    # Find best performing combinations overall
    print(f"\n" + "="*80)
    print("OVERALL BEST PERFORMERS:")
    print("="*80)

    # Best by MAE
    best_mae_idx = results_df['MAE'].idxmin()
    best_mae_row = results_df.loc[best_mae_idx]
    print(f"Best MAE: {best_mae_row['Model']} with {best_mae_row['Feature_Set']} (MAE: {best_mae_row['MAE']:.6f})")

    # Best by Scaled MAD of Normalized Bias
    best_mad_idx = results_df['Scaled_MAD_Normalized_Bias'].idxmin()
    best_mad_row = results_df.loc[best_mad_idx]
    print(f"Best Scaled MAD: {best_mad_row['Model']} with {best_mad_row['Feature_Set']} (Scaled MAD: {best_mad_row['Scaled_MAD_Normalized_Bias']:.6f})")

    # Show top 10 combinations by MAE
    print(f"\nTOP 10 MODEL/FEATURE COMBINATIONS BY MAE:")
    top_mae = results_df.sort_values('MAE').head(10)
    for i, (_, row) in enumerate(top_mae.iterrows(), 1):
        print(f"{i:2d}. {row['Model']:12s} + {row['Feature_Set']:15s} - MAE: {row['MAE']:.6f}, Scaled MAD: {row['Scaled_MAD_Normalized_Bias']:.6f}")

    # Show top 10 combinations by Scaled MAD
    print(f"\nTOP 10 MODEL/FEATURE COMBINATIONS BY SCALED MAD:")
    top_mad = results_df.sort_values('Scaled_MAD_Normalized_Bias').head(10)
    for i, (_, row) in enumerate(top_mad.iterrows(), 1):
        print(f"{i:2d}. {row['Model']:12s} + {row['Feature_Set']:15s} - Scaled MAD: {row['Scaled_MAD_Normalized_Bias']:.6f}, MAE: {row['MAE']:.6f}")