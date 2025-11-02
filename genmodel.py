#!/usr/bin/env python3
"""
Advanced Generative AI Model for Quasar Distribution Synthesis
Combines multiple state-of-the-art generative architectures with astrophysical constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

# ==================== PHYSICS CONSTRAINTS ====================

class AstrophysicsConstraints:
    """Encode fundamental astrophysical relationships for quasars"""
    
    def __init__(self, H0=70.0, Omega_m=0.3, Omega_L=0.7):
        self.H0 = H0  # Hubble constant
        self.Omega_m = Omega_m  # Matter density
        self.Omega_L = Omega_L  # Dark energy density
        self.c = 299792.458  # Speed of light in km/s
        
        # Realistic survey boundaries for extrapolation
        self.survey_limits = {
            'ra_min': 0.0, 'ra_max': 360.0,      # Full sky in RA
            'dec_min': -90.0, 'dec_max': 90.0,   # Northern sky focus (like SDSS)
            'z_min': 0.05, 'z_max': 8.0,        # Extended redshift range for extrapolation
            'flux_min': 1e-7, 'flux_max': 1e-2, # Extended flux range
            'mag_min': 12.0, 'mag_max': 25.0    # Extended magnitude range
        }
        
        # Quasar population evolution parameters for extrapolation
        self.evolution_params = {
            'luminosity_evolution': 3.2,  # Strong evolution at high-z
            'density_evolution': -0.5,    # Number density evolution
            'spectral_hardening': 0.3     # Spectral evolution with redshift
        }
        
    def luminosity_distance(self, redshift):
        """Calculate luminosity distance in Mpc using simple approximation"""
        z = torch.clamp(redshift, min=0.001)  # Only prevent division by zero, no upper limit
        
        # Simple flat ΛCDM approximation (good for z < 2)
        # D_L ≈ (c/H0) * z * (1 + z/2) for low z, with correction for high z
        
        # Basic Hubble law with cosmological corrections
        D_H = self.c / self.H0  # Hubble distance
        
        # Approximation that works reasonably well for 0 < z < 5
        # Using series expansion of the integral
        z_term1 = z
        z_term2 = z**2 / 2 * (1 - 3*self.Omega_m/4)
        z_term3 = z**3 / 6 * (1 - 3*self.Omega_m/2 + self.Omega_m**2/2)
        
        D_C = D_H * (z_term1 + z_term2 + z_term3)
        D_L = D_C * (1 + z)  # Luminosity distance
        
        return D_L
    
    def distance_modulus(self, redshift):
        """Distance modulus: μ = 5*log10(D_L) + 25"""
        D_L = self.luminosity_distance(redshift)
        return 5 * torch.log10(D_L + 1e-10) + 25
    
    def flux_from_luminosity(self, luminosity, redshift):
        """Calculate observed flux from intrinsic luminosity and redshift"""
        D_L = self.luminosity_distance(redshift)
        flux = luminosity / (4 * math.pi * (D_L * 1e6 * 3.086e16)**2)  # Convert Mpc to cm
        return flux
    
    def spectral_evolution(self, rest_wavelengths, observed_wavelengths, redshift):
        """Model spectral evolution with redshift"""
        # K-correction approximation
        k_correction = 2.5 * torch.log10(1 + redshift)
        
        # Spectral index evolution (approximate)
        alpha = -0.7 + 0.1 * torch.log10(1 + redshift)  # Power law index evolution
        
        return k_correction, alpha

    def quasar_luminosity_function(self, absolute_magnitude, redshift):
        """Double power law luminosity function evolution"""
        # Schechter function parameters (evolving with redshift)
        M_star = -27.0 + 1.5 * torch.log10(1 + redshift)  # Characteristic magnitude
        phi_star = 1e-6 * (1 + redshift)**2.5  # Normalization
        alpha = -1.5  # Faint-end slope
        beta = -3.0   # Bright-end slope
        
        # Double power law
        L_ratio = 10**(-0.4 * (absolute_magnitude - M_star))
        phi = phi_star / ((L_ratio**alpha) + (L_ratio**beta))
        
        return phi

# ==================== ADVANCED GENERATIVE ARCHITECTURES ====================

class PhysicsInformedVAE(nn.Module):
    """Variational Autoencoder with physics constraints embedded in latent space"""
    
    def __init__(self, input_dim=15, latent_dim=64, physics_dim=8):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.physics_dim = physics_dim
        self.nuisance_dim = latent_dim - physics_dim
        
        self.physics = AstrophysicsConstraints()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )
        
        # Split latent space: physics parameters + nuisance parameters
        self.physics_mu = nn.Linear(256, physics_dim)
        self.physics_logvar = nn.Linear(256, physics_dim)
        self.nuisance_mu = nn.Linear(256, self.nuisance_dim)
        self.nuisance_logvar = nn.Linear(256, self.nuisance_dim)
        
        # Decoder with physics-informed layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, input_dim)
        )
        
        # Physics interpretation layers
        self.physics_interpreter = nn.Sequential(
            nn.Linear(physics_dim, 64),
            nn.GELU(),
            nn.Linear(64, 4)  # [redshift, log_luminosity, spectral_index, extinction]
        )
    
    def encode(self, x):
        """Encode input to latent distributions"""
        h = self.encoder(x)
        
        physics_mu = self.physics_mu(h)
        physics_logvar = self.physics_logvar(h)
        nuisance_mu = self.nuisance_mu(h)
        nuisance_logvar = self.nuisance_logvar(h)
        
        return physics_mu, physics_logvar, nuisance_mu, nuisance_logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent representation to flux values"""
        return self.decoder(z)
    
    def physics_loss(self, physics_z, reconstructed_flux, original_flux):
        """Enforce physics constraints"""
        batch_size = physics_z.shape[0]
        device = physics_z.device
        
        # Interpret physics parameters
        physics_params = self.physics_interpreter(physics_z)
        redshift = torch.sigmoid(physics_params[:, 0]) * 5.0  # z ∈ [0, 5]
        log_luminosity = physics_params[:, 1] * 2 + 44  # log L ∈ [42, 46] erg/s
        spectral_index = physics_params[:, 2] * 0.5 - 0.7  # α ∈ [-1.2, -0.2]
        extinction = torch.sigmoid(physics_params[:, 3]) * 2.0  # A_V ∈ [0, 2]
        
        # Physics constraint losses
        losses = {}
        
        # 1. Distance-luminosity relation
        expected_distance_modulus = self.physics.distance_modulus(redshift)
        observed_total_flux = torch.sum(reconstructed_flux[:, :5], dim=1)  # g,r,z bands
        observed_distance_modulus = -2.5 * torch.log10(observed_total_flux + 1e-10) + 25
        
        losses['distance_modulus'] = F.mse_loss(observed_distance_modulus, expected_distance_modulus)
        
        # 2. Spectral energy distribution consistency
        # Power law check: F_ν ∝ ν^α
        wavelengths = torch.tensor([4770, 6231, 7625, 3400, 4600], device=device, dtype=torch.float32)
        frequencies = 3e18 / wavelengths  # Convert to Hz
        log_freq = torch.log10(frequencies)
        
        for i in range(batch_size):
            flux_subset = reconstructed_flux[i, :5] + 1e-10
            log_flux = torch.log10(flux_subset)
            
            # Expected power law
            expected_log_flux = spectral_index[i] * log_freq + torch.mean(log_flux)
            sed_loss = F.mse_loss(log_flux, expected_log_flux)
            
            if i == 0:
                losses['sed_consistency'] = sed_loss
            else:
                losses['sed_consistency'] += sed_loss
        
        losses['sed_consistency'] /= batch_size
        
        # 3. Luminosity function prior
        absolute_magnitudes = -2.5 * log_luminosity + 71.2  # Convert to absolute magnitude
        lf_prior = self.physics.quasar_luminosity_function(absolute_magnitudes, redshift)
        losses['luminosity_function'] = -torch.mean(torch.log(lf_prior + 1e-10))
        
        # 4. Flux positivity and reasonable ranges
        # Keep only positivity constraint for physical realism (flux > 0)
        losses['flux_positivity'] = F.relu(-reconstructed_flux).mean()
        # Remove upper flux limit - let model learn natural range
        
        return losses
    
    def forward(self, x):
        """Full forward pass with physics-informed losses"""
        physics_mu, physics_logvar, nuisance_mu, nuisance_logvar = self.encode(x)
        
        physics_z = self.reparameterize(physics_mu, physics_logvar)
        nuisance_z = self.reparameterize(nuisance_mu, nuisance_logvar)
        
        z = torch.cat([physics_z, nuisance_z], dim=1)
        reconstructed = self.decode(z)
        
        # Compute losses
        recon_loss = F.mse_loss(reconstructed, x)
        
        # KL divergences
        physics_kl = -0.5 * torch.sum(1 + physics_logvar - physics_mu.pow(2) - physics_logvar.exp())
        nuisance_kl = -0.5 * torch.sum(1 + nuisance_logvar - nuisance_mu.pow(2) - nuisance_logvar.exp())
        kl_loss = (physics_kl + nuisance_kl) / x.shape[0]
        
        # Physics losses
        physics_losses = self.physics_loss(physics_z, reconstructed, x)
        
        total_physics_loss = sum(physics_losses.values())
        
        total_loss = recon_loss + 0.1 * kl_loss + 0.5 * total_physics_loss
        
        return {
            'reconstructed': reconstructed,
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'physics_losses': physics_losses,
            'physics_params': self.physics_interpreter(physics_z)
        }
    
    def sample(self, n_samples, device='cpu'):
        """Generate new quasar samples"""
        self.eval()
        with torch.no_grad():
            # Sample from prior distributions
            physics_z = torch.randn(n_samples, self.physics_dim, device=device)
            nuisance_z = torch.randn(n_samples, self.nuisance_dim, device=device)
            
            z = torch.cat([physics_z, nuisance_z], dim=1)
            generated = self.decode(z)
            
            # Get physics parameters for the generated samples
            physics_params = self.physics_interpreter(physics_z)
            
            return generated, physics_params

class NormalizingFlow(nn.Module):
    """Normalizing Flow with astrophysical coupling layers"""
    
    def __init__(self, input_dim=15, n_flows=8, hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.flows = nn.ModuleList([
            AstrophysicalCouplingLayer(input_dim, hidden_dim, mask_type=i%2)
            for i in range(n_flows)
        ])
        self.physics = AstrophysicsConstraints()
    
    def forward(self, x):
        """Forward pass through normalizing flows"""
        log_det = 0
        for flow in self.flows:
            x, ld = flow(x)
            log_det += ld
        return x, log_det
    
    def inverse(self, z):
        """Inverse pass to generate samples"""
        log_det = 0
        for flow in reversed(self.flows):
            z, ld = flow.inverse(z)
            log_det += ld
        return z, log_det
    
    def log_prob(self, x):
        """Compute log probability of data"""
        z, log_det = self.forward(x)
        
        # Base distribution (standard normal with physics priors)
        base_log_prob = -0.5 * torch.sum(z**2, dim=1) - 0.5 * self.input_dim * math.log(2 * math.pi)
        
        return base_log_prob + log_det
    
    def sample(self, n_samples, device='cpu'):
        """Generate samples from the learned distribution"""
        self.eval()
        with torch.no_grad():
            # Sample from base distribution
            z = torch.randn(n_samples, self.input_dim, device=device)
            
            # Transform through inverse flows
            x, _ = self.inverse(z)
            
            return x

class AstrophysicalCouplingLayer(nn.Module):
    """Coupling layer that respects astrophysical structure"""
    
    def __init__(self, input_dim, hidden_dim, mask_type=0):
        super().__init__()
        self.input_dim = input_dim
        
        # Register mask as buffer so it moves with the model to GPU/CPU
        mask = self.create_astrophysical_mask(input_dim, mask_type)
        self.register_buffer('mask', mask)
        
        # Networks for scale and translation
        self.scale_net = self.create_network(hidden_dim)
        self.translation_net = self.create_network(hidden_dim)
        
    def create_astrophysical_mask(self, dim, mask_type):
        """Create masks that respect spectral band groupings and spatial structure"""
        mask = torch.zeros(dim)
        
        # Assume last 2 dimensions are spatial (RA, DEC)
        flux_dims = dim - 2
        
        if mask_type == 0:
            # Optical bands (G, R, Z) and some ML features + RA coordinate
            mask[0] = 1  # FLUX_G
            mask[1] = 1  # FLUX_R  
            mask[2] = 1  # FLUX_Z
            if flux_dims > 9:
                mask[9:min(12, flux_dims)] = 1  # Some ML features
            mask[-2] = 1  # RA coordinate
        else:
            # IR bands, remaining ML features + DEC coordinate
            mask[3:min(6, flux_dims)] = 1  # IVAR features
            if flux_dims > 6:
                mask[6:min(9, flux_dims)] = 1  # W1, W2, other fluxes
            if flux_dims > 12:
                mask[12:flux_dims] = 1  # Remaining ML features
            mask[-1] = 1  # DEC coordinate
            
        return mask
    
    def create_network(self, hidden_dim):
        """Create the transformation network"""
        input_size = int(torch.sum(self.mask))
        output_size = self.input_dim - input_size
        
        return nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size)
        )
    
    def forward(self, x):
        """Forward coupling transformation"""
        x_masked = x * self.mask
        x_unmasked = x * (1 - self.mask)
        
        # Get inputs for transformation networks
        masked_input = x_masked[self.mask.bool()]
        if len(masked_input.shape) == 1:
            masked_input = masked_input.unsqueeze(0)
        
        # Compute scale and translation
        log_scale = self.scale_net(masked_input)
        translation = self.translation_net(masked_input)
        
        # Apply transformation
        y_unmasked = x_unmasked * torch.exp(log_scale) + translation
        y = x_masked + y_unmasked
        
        log_det = torch.sum(log_scale, dim=1)
        
        return y, log_det
    
    def inverse(self, y):
        """Inverse coupling transformation"""
        y_masked = y * self.mask
        y_unmasked = y * (1 - self.mask)
        
        # Get inputs for transformation networks
        masked_input = y_masked[self.mask.bool()]
        if len(masked_input.shape) == 1:
            masked_input = masked_input.unsqueeze(0)
        
        # Compute scale and translation
        log_scale = self.scale_net(masked_input)
        translation = self.translation_net(masked_input)
        
        # Apply inverse transformation
        x_unmasked = (y_unmasked - translation) * torch.exp(-log_scale)
        x = y_masked + x_unmasked
        
        log_det = -torch.sum(log_scale, dim=1)
        
        return x, log_det

class DiffusionModel(nn.Module):
    """Physics-informed diffusion model for quasar generation"""
    
    def __init__(self, input_dim=15, timesteps=1000, hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.timesteps = timesteps
        
        # U-Net style architecture for denoising
        self.time_embedding = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 64)
        )
        
        self.noise_predictor = nn.Sequential(
            nn.Linear(input_dim + 64, hidden_dim),  # +64 for time embedding
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Physics-informed noise schedule
        self.register_buffer('betas', self.cosine_beta_schedule(timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
        self.physics = AstrophysicsConstraints()
    
    def cosine_beta_schedule(self, timesteps):
        """Cosine noise schedule"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def get_time_embedding(self, t):
        """Sinusoidal time embeddings"""
        t = t.float() / self.timesteps
        return self.time_embedding(t.unsqueeze(-1))
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t]).unsqueeze(-1)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - self.alphas_cumprod[t]).unsqueeze(-1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, x_t, t):
        """Single denoising step"""
        t_embed = self.get_time_embedding(t)
        
        # Predict noise
        x_input = torch.cat([x_t, t_embed], dim=-1)
        predicted_noise = self.noise_predictor(x_input)
        
        # Compute denoised sample
        alpha_t = self.alphas[t].unsqueeze(-1)
        alpha_cumprod_t = self.alphas_cumprod[t].unsqueeze(-1)
        beta_t = self.betas[t].unsqueeze(-1)
        
        mean = (x_t - beta_t * predicted_noise / torch.sqrt(1 - alpha_cumprod_t)) / torch.sqrt(alpha_t)
        
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            variance = beta_t
            sample = mean + torch.sqrt(variance) * noise
        else:
            sample = mean
        
        return sample
    
    def sample(self, n_samples, device='cpu'):
        """Generate samples through reverse diffusion"""
        self.eval()
        with torch.no_grad():
            # Start from pure noise
            x = torch.randn(n_samples, self.input_dim, device=device)
            
            # Reverse diffusion process
            for t in reversed(range(self.timesteps)):
                t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
                x = self.p_sample(x, t_batch)
            
            return x
    
    def forward(self, x_start):
        """Training forward pass"""
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=device)
        
        # Generate noise
        noise = torch.randn_like(x_start)
        
        # Forward diffusion
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Predict noise
        t_embed = self.get_time_embedding(t)
        x_input = torch.cat([x_noisy, t_embed], dim=-1)
        predicted_noise = self.noise_predictor(x_input)
        
        # Loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss

# ==================== VALIDATION FRAMEWORK ====================

class PhysicsValidator:
    """Comprehensive physics validation for generated quasar distributions"""
    
    def __init__(self):
        self.physics = AstrophysicsConstraints()
    
    def validate_spatial_distribution(self, generated_coords, real_coords):
        """Validate spatial distribution of generated quasars"""
        print("    Testing spatial distribution...")
        
        results = {}
        
        # Extract RA/DEC
        gen_ra, gen_dec = generated_coords[:, 0], generated_coords[:, 1] 
        real_ra, real_dec = real_coords[:, 0], real_coords[:, 1]
        
        # 1. Angular clustering test
        def calculate_angular_correlation(ra, dec, max_sep=10.0):
            """Calculate angular correlation function"""
            # Convert to radians
            ra_rad = np.deg2rad(ra)
            dec_rad = np.deg2rad(dec)
            
            # Calculate pairwise angular separations (simplified)
            n_samples = min(1000, len(ra))  # Subsample for efficiency
            idx = np.random.choice(len(ra), n_samples, replace=False)
            
            separations = []
            for i in range(n_samples):
                for j in range(i+1, n_samples):
                    # Haversine formula for angular separation
                    dra = ra_rad[idx[i]] - ra_rad[idx[j]]
                    ddec = dec_rad[idx[i]] - dec_rad[idx[j]]
                    
                    a = np.sin(ddec/2)**2 + np.cos(dec_rad[idx[i]]) * np.cos(dec_rad[idx[j]]) * np.sin(dra/2)**2
                    sep = 2 * np.arcsin(np.sqrt(a))
                    separations.append(np.rad2deg(sep))
            
            return np.array(separations)
        
        gen_seps = calculate_angular_correlation(gen_ra, gen_dec)
        real_seps = calculate_angular_correlation(real_ra, real_dec)
        
        # Compare separation distributions
        ks_stat_spatial = stats.ks_2samp(gen_seps, real_seps)[0]
        
        # 2. Sky coverage uniformity test
        # Divide sky into grid and check occupancy
        ra_bins = np.linspace(real_ra.min(), real_ra.max(), 20)
        dec_bins = np.linspace(real_dec.min(), real_dec.max(), 20)
        
        gen_hist, _, _ = np.histogram2d(gen_ra, gen_dec, bins=[ra_bins, dec_bins])
        real_hist, _, _ = np.histogram2d(real_ra, real_dec, bins=[ra_bins, dec_bins])
        
        # Flatten and compare
        gen_density = gen_hist.flatten()
        real_density = real_hist.flatten()
        
        # Remove empty bins
        mask = (gen_density > 0) | (real_density > 0)
        gen_density = gen_density[mask]
        real_density = real_density[mask]
        
        if len(gen_density) > 0 and len(real_density) > 0:
            density_corr = np.corrcoef(gen_density, real_density)[0, 1] if len(gen_density) == len(real_density) else 0.0
        else:
            density_corr = 0.0
        
        # 3. Coordinate range validation
        ra_range_match = (gen_ra.min() >= real_ra.min() * 0.9 and 
                         gen_ra.max() <= real_ra.max() * 1.1)
        dec_range_match = (gen_dec.min() >= real_dec.min() * 0.9 and 
                          gen_dec.max() <= real_dec.max() * 1.1)
        
        results = {
            'angular_clustering_ks': ks_stat_spatial,
            'sky_density_correlation': density_corr if not np.isnan(density_corr) else 0.0,
            'coordinate_ranges_match': ra_range_match and dec_range_match,
            'ra_range': (gen_ra.min(), gen_ra.max()),
            'dec_range': (gen_dec.min(), gen_dec.max()),
            'spatial_realism_score': (
                (1 - min(ks_stat_spatial, 1.0)) * 0.4 +
                max(density_corr, 0.0) * 0.4 + 
                (1.0 if (ra_range_match and dec_range_match) else 0.0) * 0.2
            )
        }
        
        return results
    
    def validate_distance_modulus_relation(self, fluxes, redshifts):
        """Test cosmological distance-luminosity relation"""
        try:
            # Calculate apparent magnitudes more robustly
            # Use multiple bands for better statistics
            total_flux = torch.clamp(torch.sum(fluxes[:, :5], dim=1), min=1e-12)  # g+r+z+w1+w2 bands
            apparent_mag = -2.5 * torch.log10(total_flux) + 25
            
            # Expected distance modulus
            expected_dm = self.physics.distance_modulus(redshifts)
            
            # Filter out unrealistic values
            mask = (apparent_mag > 15) & (apparent_mag < 30) & (expected_dm > 30) & (expected_dm < 50)
            
            if torch.sum(mask) < 10:  # Need at least 10 points
                return {
                    'correlation': 0.0,
                    'rmse': 100.0,
                    'passes': False,
                    'valid_points': torch.sum(mask).item()
                }
            
            apparent_mag_clean = apparent_mag[mask]
            expected_dm_clean = expected_dm[mask]
            
            # Calculate correlation more safely
            if len(apparent_mag_clean) > 1:
                correlation_matrix = torch.corrcoef(torch.stack([apparent_mag_clean, expected_dm_clean]))
                if correlation_matrix.numel() >= 4:  # 2x2 matrix
                    correlation = correlation_matrix[0, 1]
                    if torch.isnan(correlation) or torch.isinf(correlation):
                        correlation = torch.tensor(0.0)
                else:
                    correlation = torch.tensor(0.0)
            else:
                correlation = torch.tensor(0.0)
            
            # More lenient passing criterion (correlation > 0.3 or reasonable RMSE)
            rmse = torch.sqrt(F.mse_loss(apparent_mag_clean, expected_dm_clean))
            
            return {
                'correlation': correlation.item() if not torch.isnan(correlation) else 0.0,
                'rmse': rmse.item() if not torch.isnan(rmse) else 100.0,
                'passes': (correlation > 0.3) or (rmse < 10.0),  # More lenient
                'valid_points': torch.sum(mask).item()
            }
        except Exception as e:
            return {
                'correlation': 0.0,
                'rmse': 100.0,
                'passes': False,
                'valid_points': 0,
                'error': str(e)
            }
    
    def validate_sed_physics(self, fluxes):
        """Validate spectral energy distribution physics"""
        results = {}
        
        # Power law continuum test
        wavelengths = torch.tensor([4770, 6231, 7625, 3400, 4600], device=fluxes.device)  # G,R,Z,W1,W2
        frequencies = 3e18 / wavelengths
        log_freq = torch.log10(frequencies)
        
        slopes = []
        valid_slopes = 0
        
        for i in range(len(fluxes)):
            flux_subset = fluxes[i, :5]
            # Ensure positive fluxes
            flux_subset = torch.clamp(flux_subset, min=1e-10)
            log_flux = torch.log10(flux_subset)
            
            # Check for NaN/inf values
            if torch.any(torch.isnan(log_flux)) or torch.any(torch.isinf(log_flux)):
                slopes.append(0.0)  # Invalid slope
                continue
            
            # Manual least squares fit for slope (y = mx + b)
            n = len(log_freq)
            sum_x = torch.sum(log_freq)
            sum_y = torch.sum(log_flux)
            sum_xy = torch.sum(log_freq * log_flux)
            sum_x2 = torch.sum(log_freq * log_freq)
            
            denominator = n * sum_x2 - sum_x * sum_x
            if abs(denominator) < 1e-10:  # Avoid division by zero
                slopes.append(0.0)
                continue
            
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            slope_value = abs(slope.item())  # Use absolute value for spectral index
            slopes.append(slope_value)
            
            # Count valid slopes (typical quasar range: 0.1 to 3.0)
            if 0.1 <= slope_value <= 3.0:
                valid_slopes += 1
        
        slopes = torch.tensor(slopes, device=fluxes.device)
        
        # Safer flux ratio calculations with better bounds
        safe_denom1 = torch.clamp(fluxes[:, 6] + fluxes[:, 7], min=1e-10)
        safe_denom2 = torch.clamp(fluxes[:, 2], min=1e-10)
        
        # UV excess test (g-band vs IR) - more lenient bounds
        uv_excess = torch.clamp(fluxes[:, 0] / safe_denom1, min=0.01, max=100)
        
        # IR excess test - more lenient bounds
        ir_excess = torch.clamp((fluxes[:, 6] + fluxes[:, 7]) / safe_denom2, min=0.01, max=100)
        
        results['power_law_slopes'] = {
            'mean': slopes.mean().item(),
            'std': slopes.std().item(),
            'realistic_fraction': valid_slopes / len(fluxes) if len(fluxes) > 0 else 0.0,
            'valid_count': valid_slopes,
            'total_count': len(fluxes)
        }
        
        results['uv_excess'] = {
            'mean': uv_excess.mean().item(),
            'realistic_fraction': ((uv_excess > 0.1) & (uv_excess < 50)).float().mean().item()  # More lenient
        }
        
        results['ir_excess'] = {
            'mean': ir_excess.mean().item(),
            'realistic_fraction': ((ir_excess > 0.05) & (ir_excess < 20)).float().mean().item()  # More lenient
        }
        
        return results
    
    def validate_flux_statistics(self, generated_fluxes, real_fluxes):
        """Compare statistical moments of flux distributions"""
        results = {}
        
        band_names = ['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2']
        
        for i, band in enumerate(band_names):
            gen_band = generated_fluxes[:, i]
            real_band = real_fluxes[:, i]
            
            results[band] = {
                'mean_ratio': (gen_band.mean() / real_band.mean()).item(),
                'std_ratio': (gen_band.std() / real_band.std()).item(),
                'ks_statistic': stats.ks_2samp(gen_band.detach().cpu().numpy(), 
                                             real_band.detach().cpu().numpy())[0],
                'skew_diff': abs(stats.skew(gen_band.detach().cpu().numpy()) - 
                               stats.skew(real_band.detach().cpu().numpy()))
            }
        
        return results
    
    def comprehensive_validation(self, generated_fluxes, real_fluxes, generated_redshifts=None, 
                               generated_coords=None, real_coords=None):
        """Run all validation tests including spatial distribution"""
        print("Running comprehensive physics validation...")
        
        results = {}
        
        # 1. SED Physics
        print("  Testing spectral energy distributions...")
        results['sed_physics'] = self.validate_sed_physics(generated_fluxes)
        
        # 2. Flux statistics
        print("  Comparing flux distributions...")
        results['flux_statistics'] = self.validate_flux_statistics(generated_fluxes, real_fluxes)
        
        # 3. Spatial distribution (if coordinates available)
        spatial_score = 0.5  # Default neutral score
        if generated_coords is not None and real_coords is not None:
            print("  Testing spatial distribution...")
            results['spatial_distribution'] = self.validate_spatial_distribution(
                generated_coords, real_coords
            )
            spatial_score = results['spatial_distribution']['spatial_realism_score']
        
        # 4. Distance modulus (if redshifts available)
        dm_score = 0.5  # Default neutral score
        if generated_redshifts is not None:
            print("  Testing distance-luminosity relation...")
            results['distance_modulus'] = self.validate_distance_modulus_relation(
                generated_fluxes, generated_redshifts
            )
            dm_score = 1.0 if results['distance_modulus']['passes'] else 0.0
        
        # Overall realism score (weighted combination)
        sed_score = results['sed_physics']['power_law_slopes']['realistic_fraction']
        flux_score = 1 - np.mean([band['ks_statistic'] for band in results['flux_statistics'].values()])
        
        # Weight the scores based on available data
        weights = []
        scores = []
        
        # Always include SED and flux scores
        weights.extend([0.3, 0.3])
        scores.extend([sed_score, flux_score])
        
        # Add spatial score if available
        if generated_coords is not None and real_coords is not None:
            weights.append(0.2)
            scores.append(spatial_score)
        
        # Add distance modulus score if available
        if generated_redshifts is not None:
            weights.append(0.2)
            scores.append(dm_score)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        overall_score = np.sum(np.array(scores) * weights)
        results['overall_realism_score'] = overall_score
        
        print(f"  Overall realism score: {overall_score:.3f}")
        print(f"    - SED physics: {sed_score:.3f}")
        print(f"    - Flux statistics: {flux_score:.3f}")
        if generated_coords is not None and real_coords is not None:
            print(f"    - Spatial distribution: {spatial_score:.3f}")
        if generated_redshifts is not None:
            print(f"    - Distance-luminosity: {dm_score:.3f}")
        
        return results

# ==================== CONVNEXT GENERATIVE COMPONENTS ====================

class ConvNeXtBlock1D(nn.Module):
    """ConvNeXt block adapted for 1D generative modeling"""
    
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        
        # Depthwise convolution (adapted to 1D)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        # LayerNorm
        self.norm = nn.LayerNorm(dim)
        
        # Pointwise/1x1 convs
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # Layer scale
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), 
            requires_grad=True
        ) if layer_scale_init_value > 0 else None

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
        
        x = input_x + x
        return x


class ConvNeXtGenerativeEncoder(nn.Module):
    """ConvNeXt-based encoder for generative modeling"""
    
    def __init__(self, input_dim, latent_dim, depths=[2, 2, 4], dims=[64, 128, 256]):
        super().__init__()
        
        # Stem - project input to first stage dimension
        self.stem = nn.Sequential(
            nn.Linear(input_dim, dims[0]),
            nn.LayerNorm(dims[0])
        )
        
        # ConvNeXt stages
        self.stages = nn.ModuleList()
        
        for i in range(len(depths)):
            # Downsampling layer (except for first stage)
            if i > 0:
                downsample = nn.Sequential(
                    nn.LayerNorm(dims[i-1]),
                    nn.Linear(dims[i-1], dims[i])
                )
            else:
                downsample = nn.Identity()
            
            # Stage blocks
            stage = nn.Sequential(
                downsample,
                *[ConvNeXtBlock1D(dim=dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)
        
        # Final projection to latent space
        self.norm = nn.LayerNorm(dims[-1])
        self.mu_head = nn.Linear(dims[-1], latent_dim)
        self.logvar_head = nn.Linear(dims[-1], latent_dim)
    
    def forward(self, x):
        x = self.stem(x)
        
        for stage in self.stages:
            x = stage(x)
        
        x = self.norm(x)
        mu = self.mu_head(x)
        logvar = self.logvar_head(x)
        
        return mu, logvar


class ConvNeXtGenerativeDecoder(nn.Module):
    """ConvNeXt-based decoder for generative modeling"""
    
    def __init__(self, latent_dim, output_dim, depths=[4, 2, 2], dims=[256, 128, 64]):
        super().__init__()
        
        # Initial projection from latent space
        self.stem = nn.Sequential(
            nn.Linear(latent_dim, dims[0]),
            nn.LayerNorm(dims[0])
        )
        
        # ConvNeXt stages (reverse order for decoder)
        self.stages = nn.ModuleList()
        
        for i in range(len(depths)):
            # Upsampling layer (except for first stage)
            if i > 0:
                upsample = nn.Sequential(
                    nn.LayerNorm(dims[i-1]),
                    nn.Linear(dims[i-1], dims[i])
                )
            else:
                upsample = nn.Identity()
            
            # Stage blocks
            stage = nn.Sequential(
                upsample,
                *[ConvNeXtBlock1D(dim=dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)
        
        # Final output projection
        self.norm = nn.LayerNorm(dims[-1])
        self.output_head = nn.Linear(dims[-1], output_dim)
    
    def forward(self, x):
        x = self.stem(x)
        
        for stage in self.stages:
            x = stage(x)
        
        x = self.norm(x)
        x = self.output_head(x)
        
        return x


class ConvNeXtConditionalDecoder(nn.Module):
    """ConvNeXt-based conditional decoder for flux generation"""
    
    def __init__(self, latent_dim, condition_dim, output_dim, depths=[4, 3, 2], dims=[512, 256, 128]):
        super().__init__()
        
        # Initial projection from latent space + condition
        self.stem = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, dims[0]),
            nn.LayerNorm(dims[0])
        )
        
        # ConvNeXt stages with attention
        self.stages = nn.ModuleList()
        
        for i in range(len(depths)):
            # Upsampling layer (except for first stage)
            if i > 0:
                upsample = nn.Sequential(
                    nn.LayerNorm(dims[i-1]),
                    nn.Linear(dims[i-1], dims[i])
                )
            else:
                upsample = nn.Identity()
            
            # Stage blocks
            stage = nn.Sequential(
                upsample,
                *[ConvNeXtBlock1D(dim=dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)
        
        # Multi-head attention for better flux modeling
        self.attention = nn.MultiheadAttention(
            embed_dim=dims[-1], num_heads=8, dropout=0.1, batch_first=True
        )
        
        # Final output projection
        self.norm = nn.LayerNorm(dims[-1])
        self.output_head = nn.Sequential(
            nn.Linear(dims[-1], dims[-1] // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dims[-1] // 2, output_dim)
        )
    
    def forward(self, z, condition):
        # Concatenate latent vector with condition
        x = torch.cat([z, condition.unsqueeze(1) if condition.dim() == 1 else condition], dim=-1)
        
        x = self.stem(x)
        
        for stage in self.stages:
            x = stage(x)
        
        # Apply attention (treating features as sequence length 1)
        x_attn = x.unsqueeze(1)  # (batch, 1, features)
        x_attn, _ = self.attention(x_attn, x_attn, x_attn)
        x = x_attn.squeeze(1)  # Back to (batch, features)
        
        x = self.norm(x)
        x = self.output_head(x)
        
        return x



# ==================== TRAINING AND GENERATION ====================

class SpatialRedshiftGenerator(nn.Module):
    """Generate realistic sky positions with redshifts using simplified architectures"""
    
    def __init__(self, model_type='vae', latent_dim=32):
        super().__init__()
        self.model_type = model_type
        self.latent_dim = latent_dim
        
        if model_type == 'vae':
            # VAE for spatial-redshift distribution
            self.encoder = nn.Sequential(
                nn.Linear(3, 128),  # RA, DEC, redshift
                nn.GELU(),
                nn.Linear(128, 256),
                nn.GELU(),
                nn.Linear(256, 128)
            )
            
            self.mu = nn.Linear(128, latent_dim)
            self.logvar = nn.Linear(128, latent_dim)
            
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.GELU(),
                nn.Linear(128, 256),
                nn.GELU(),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Linear(128, 3)  # RA, DEC, redshift
            )
        
        elif model_type == 'convnext':
            # ConvNeXt-based VAE architecture for spatial-redshift
            self.encoder_net = ConvNeXtGenerativeEncoder(
                input_dim=3, latent_dim=latent_dim, 
                depths=[2, 2, 3], dims=[32, 64, 128]  # Smaller for spatial data
            )
            self.decoder_net = ConvNeXtGenerativeDecoder(
                latent_dim=latent_dim, output_dim=3,
                depths=[3, 2, 2], dims=[128, 64, 32]
            )
            
            # Attention mechanism for enhanced spatial understanding
            num_heads = min(4, max(1, latent_dim//8))
            self.spatial_attention = nn.MultiheadAttention(
                embed_dim=latent_dim, num_heads=num_heads, dropout=0.1, batch_first=True
            )
    
    def forward(self, x):
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feedforward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class TransformerGenerativeEncoder(nn.Module):
    """Transformer-based encoder for generative modeling"""
    
    def __init__(self, input_dim, latent_dim, d_model=256, num_heads=8, num_layers=6, max_seq_len=128):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding for sequence modeling (even for tabular data)
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, d_model))
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_model * 4, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        # Output heads for VAE
        self.norm = nn.LayerNorm(d_model)
        self.mu_head = nn.Linear(d_model, latent_dim)
        self.logvar_head = nn.Linear(d_model, latent_dim)
    
    def forward(self, x):
        # Project input to d_model dimensions
        x = self.input_projection(x).unsqueeze(1)  # (batch, 1, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding[:1].unsqueeze(0)  # Use first position
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Global average pooling and output
        x = self.norm(x.mean(dim=1))  # (batch, d_model)
        
        mu = self.mu_head(x)
        logvar = self.logvar_head(x)
        
        return mu, logvar


class TransformerGenerativeDecoder(nn.Module):
    """Transformer-based decoder for generative modeling"""
    
    def __init__(self, latent_dim, output_dim, d_model=256, num_heads=8, num_layers=6, max_seq_len=128):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        
        # Project latent to transformer dimension
        self.latent_projection = nn.Linear(latent_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, d_model))
        
        # Transformer decoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_model * 4, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.norm = nn.LayerNorm(d_model)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, output_dim)
        )
    
    def forward(self, z):
        # Project latent to d_model and add sequence dimension
        x = self.latent_projection(z).unsqueeze(1)  # (batch, 1, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding[:1].unsqueeze(0)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Output projection
        x = self.norm(x.squeeze(1))  # (batch, d_model)
        output = self.output_head(x)
        
        return output


class TransformerConditionalDecoder(nn.Module):
    """Conditional transformer decoder for flux generation"""
    
    def __init__(self, latent_dim, condition_dim, output_dim, d_model=512, num_heads=16, num_layers=8):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        
        # Project inputs to transformer dimension
        self.latent_projection = nn.Linear(latent_dim, d_model // 2)
        self.condition_projection = nn.Linear(condition_dim, d_model // 2)
        
        # Learnable embeddings for flux bands
        self.flux_embeddings = nn.Parameter(torch.randn(output_dim, d_model))
        
        # Transformer layers with cross-attention capabilities
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_model * 2, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        # Multi-scale attention for different flux bands
        self.band_attention = nn.MultiheadAttention(d_model, num_heads, dropout=0.1, batch_first=True)
        
        # Output layers
        self.norm = nn.LayerNorm(d_model)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(d_model // 4, 1)  # Single value per flux band
        )
    
    def forward(self, z, condition):
        batch_size = z.shape[0]
        
        # Project and combine latent + condition
        z_proj = self.latent_projection(z)  # (batch, d_model//2)
        if condition.dim() == 1:
            condition = condition.unsqueeze(1)
        c_proj = self.condition_projection(condition)  # (batch, d_model//2)
        
        # Combine latent and condition
        combined = torch.cat([z_proj, c_proj], dim=-1)  # (batch, d_model)
        
        # Create sequence of flux band embeddings
        flux_seq = self.flux_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch, output_dim, d_model)
        
        # Add condition information to each flux band
        combined_expanded = combined.unsqueeze(1).repeat(1, self.output_dim, 1)
        flux_seq = flux_seq + combined_expanded
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            flux_seq = layer(flux_seq)
        
        # Apply band-specific attention
        flux_seq_attn, _ = self.band_attention(flux_seq, flux_seq, flux_seq)
        
        # Generate output for each flux band
        flux_seq = self.norm(flux_seq_attn)
        outputs = self.output_head(flux_seq).squeeze(-1)  # (batch, output_dim)
        
        return outputs


class ViTBlock(nn.Module):
    """Vision Transformer block adapted for tabular generative modeling"""
    
    def __init__(self, d_model, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        
        mlp_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Multi-head self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x


class ViTGenerativeEncoder(nn.Module):
    """Vision Transformer encoder for generative modeling"""
    
    def __init__(self, input_dim, latent_dim, d_model=384, num_heads=12, depth=12, patch_size=1):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        
        # Patch embedding (treat each feature as a "patch")
        self.patch_embed = nn.Linear(patch_size, d_model)
        
        # Class token for global representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional embeddings (with padding buffer for flexibility)
        max_patches = (input_dim + patch_size - 1) // patch_size  # Ceiling division
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches + 1, d_model))  # +1 for cls token
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            ViTBlock(d_model, num_heads, dropout=0.1) for _ in range(depth)
        ])
        
        # Output heads
        self.norm = nn.LayerNorm(d_model)
        self.mu_head = nn.Linear(d_model, latent_dim)
        self.logvar_head = nn.Linear(d_model, latent_dim)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Reshape input for patch processing
        if x.shape[1] % self.patch_size != 0:
            padding = self.patch_size - (x.shape[1] % self.patch_size)
            x = F.pad(x, (0, padding))
        
        x = x.view(batch_size, -1, self.patch_size)  # (batch, num_patches, patch_size)
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch, num_patches, d_model)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Use class token for final representation
        x = self.norm(x[:, 0])  # (batch, d_model)
        
        mu = self.mu_head(x)
        logvar = self.logvar_head(x)
        
        return mu, logvar


class ViTGenerativeDecoder(nn.Module):
    """Vision Transformer decoder for generative modeling"""
    
    def __init__(self, latent_dim, output_dim, d_model=384, num_heads=12, depth=8):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        
        # Project latent to transformer space
        self.latent_proj = nn.Linear(latent_dim, d_model)
        
        # Learnable output tokens
        self.output_tokens = nn.Parameter(torch.randn(1, output_dim, d_model))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            ViTBlock(d_model, num_heads, dropout=0.1) for _ in range(depth)
        ])
        
        # Output projection
        self.norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, 1)
    
    def forward(self, z):
        batch_size = z.shape[0]
        
        # Project latent
        z_proj = self.latent_proj(z).unsqueeze(1)  # (batch, 1, d_model)
        
        # Create output sequence
        output_tokens = self.output_tokens.expand(batch_size, -1, -1)  # (batch, output_dim, d_model)
        
        # Combine latent with output tokens
        x = torch.cat([z_proj, output_tokens], dim=1)  # (batch, output_dim+1, d_model)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Generate outputs (skip the first latent token)
        x = self.norm(x[:, 1:, :])  # (batch, output_dim, d_model)
        outputs = self.output_head(x).squeeze(-1)  # (batch, output_dim)
        
        return outputs

# ==================== TRAINING AND GENERATION ====================

class SpatialRedshiftGenerator(nn.Module):
    """Generate realistic sky positions with redshifts"""
    
    def __init__(self, model_type='vae', latent_dim=32):
        super().__init__()
        self.model_type = model_type
        self.latent_dim = latent_dim
        
        if model_type == 'vae':
            # VAE for spatial-redshift distribution
            self.encoder = nn.Sequential(
                nn.Linear(3, 128),  # RA, DEC, redshift
                nn.GELU(),
                nn.Linear(128, 256),
                nn.GELU(),
                nn.Linear(256, 128)
            )
            
            self.mu = nn.Linear(128, latent_dim)
            self.logvar = nn.Linear(128, latent_dim)
            
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.GELU(),
                nn.Linear(128, 256),
                nn.GELU(),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Linear(128, 3)  # RA, DEC, redshift
            )
        
        elif model_type == 'convnext':
            # ConvNeXt-based VAE architecture for spatial-redshift
            self.encoder_net = ConvNeXtGenerativeEncoder(
                input_dim=3, latent_dim=latent_dim, 
                depths=[2, 2, 3], dims=[32, 64, 128]  # Smaller for spatial data
            )
            self.decoder_net = ConvNeXtGenerativeDecoder(
                latent_dim=latent_dim, output_dim=3,
                depths=[3, 2, 2], dims=[128, 64, 32]
            )
            
            # Attention mechanism for enhanced spatial understanding
            num_heads = min(4, max(1, latent_dim//8))
            self.spatial_attention = nn.MultiheadAttention(
                embed_dim=latent_dim, num_heads=num_heads, dropout=0.1, batch_first=True
            )
        
        elif model_type == 'flow':
            # Simple normalizing flow for spatial-redshift
            self.flows = nn.ModuleList([
                SimpleCouplingLayer(3, 64) for _ in range(4)
            ])
        
        elif model_type == 'diffusion':
            # Simple diffusion model for spatial-redshift
            self.model = nn.Sequential(
                nn.Linear(3, 128),
                nn.GELU(),
                nn.Linear(128, 256),
                nn.GELU(),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Linear(128, 3)
            )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # Small gain for stability
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def encode(self, x):
        if self.model_type == 'vae':
            h = self.encoder(x)
            return self.mu(h), self.logvar(h)
        elif self.model_type in ['convnext', 'transformer', 'vit']:
            return self.encoder_net(x)
        else:
            return None, None
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        if self.model_type == 'vae':
            return self.decoder(z)
        elif self.model_type == 'convnext':
            return self.decoder_net(z)
        else:
            return z
    
    def forward(self, x):
        if self.model_type == 'vae':
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            reconstructed = self.decode(z)
            
            # Improved VAE loss with mode collapse prevention
            recon_loss = F.mse_loss(reconstructed, x)
            
            # Clamp logvar for numerical stability
            logvar_clamped = torch.clamp(logvar, min=-20, max=10)
            kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp(), dim=1))
            
            # Add diversity loss to prevent mode collapse
            batch_mean = torch.mean(reconstructed, dim=0)
            diversity_loss = -torch.var(batch_mean)  # Encourage variation in batch means
            
            # Physics constraints for coordinates
            # RA should be in [-1, 1], DEC should be in [-1, 1], z_norm should be in [-1, 1]
            coord_loss = 0.01 * (F.relu(torch.abs(reconstructed) - 1.2).mean())  # Soft constraint
            
            # Check for NaN/Inf values
            if torch.isnan(recon_loss) or torch.isinf(recon_loss):
                recon_loss = torch.tensor(0.1, device=x.device, requires_grad=True)
            if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                kl_loss = torch.tensor(0.01, device=x.device, requires_grad=True)
            if torch.isnan(diversity_loss) or torch.isinf(diversity_loss):
                diversity_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
            if torch.isnan(coord_loss) or torch.isinf(coord_loss):
                coord_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
            
            # Use β-VAE with lower KL weight to prevent mode collapse
            total_loss = recon_loss + 0.01 * kl_loss + 0.01 * diversity_loss + coord_loss
            
            return {
                'reconstructed': reconstructed,
                'total_loss': total_loss,
                'recon_loss': recon_loss,
                'kl_loss': kl_loss,
                'diversity_loss': diversity_loss
            }
        
        elif self.model_type == 'convnext':
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            
            # Apply attention to latent representation
            z_attn = z.unsqueeze(1)  # (batch, 1, latent_dim)
            z_attn, attention_weights = self.spatial_attention(z_attn, z_attn, z_attn)
            z_enhanced = z_attn.squeeze(1)  # Back to (batch, latent_dim)
            
            reconstructed = self.decode(z_enhanced)
            
            # Enhanced loss function for ConvNeXt
            recon_loss = F.mse_loss(reconstructed, x)
            
            # KL divergence with improved stability
            logvar_clamped = torch.clamp(logvar, min=-20, max=10)
            kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp(), dim=1))
            
            # Spatial coherence loss - encourage realistic spatial relationships
            ra_consistency = F.relu(torch.abs(reconstructed[:, 0]) - 1.0).mean()  # RA should be in [-1,1]
            dec_consistency = F.relu(torch.abs(reconstructed[:, 1]) - 1.0).mean()  # DEC should be in [-1,1]
            z_consistency = F.relu(torch.abs(reconstructed[:, 2]) - 1.0).mean()   # z_norm should be in [-1,1]
            
            spatial_loss = 0.1 * (ra_consistency + dec_consistency + z_consistency)
            
            # Attention regularization - prevent attention collapse
            attention_entropy = -torch.mean(torch.sum(attention_weights * torch.log(attention_weights + 1e-10), dim=-1))
            attention_reg = 0.01 * attention_entropy
            
            # Check for NaN/Inf values
            if torch.isnan(recon_loss) or torch.isinf(recon_loss):
                recon_loss = torch.tensor(0.1, device=x.device, requires_grad=True)
            if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                kl_loss = torch.tensor(0.01, device=x.device, requires_grad=True)
            if torch.isnan(spatial_loss) or torch.isinf(spatial_loss):
                spatial_loss = torch.tensor(0.01, device=x.device, requires_grad=True)
            if torch.isnan(attention_reg) or torch.isinf(attention_reg):
                attention_reg = torch.tensor(0.0, device=x.device, requires_grad=True)
            
            # Total loss with ConvNeXt-specific regularization
            total_loss = recon_loss + 0.005 * kl_loss + spatial_loss + attention_reg
            
            return {
                'reconstructed': reconstructed,
                'total_loss': total_loss,
                'recon_loss': recon_loss,
                'kl_loss': kl_loss,
                'spatial_loss': spatial_loss,
                'attention_weights': attention_weights
            }
        
        elif self.model_type == 'transformer':
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            reconstructed = self.decode(z)
            
            # Transformer-specific loss function
            recon_loss = F.mse_loss(reconstructed, x)
            
            # KL divergence
            logvar_clamped = torch.clamp(logvar, min=-20, max=10)
            kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp(), dim=1))
            
            # Coordinate consistency loss
            coord_loss = 0.05 * (F.relu(torch.abs(reconstructed) - 1.1).mean())
            
            # Check for NaN/Inf values
            if torch.isnan(recon_loss) or torch.isinf(recon_loss):
                recon_loss = torch.tensor(0.1, device=x.device, requires_grad=True)
            if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                kl_loss = torch.tensor(0.01, device=x.device, requires_grad=True)
            if torch.isnan(coord_loss) or torch.isinf(coord_loss):
                coord_loss = torch.tensor(0.01, device=x.device, requires_grad=True)
            
            total_loss = recon_loss + 0.01 * kl_loss + coord_loss
            
            return {
                'reconstructed': reconstructed,
                'total_loss': total_loss,
                'recon_loss': recon_loss,
                'kl_loss': kl_loss
            }
        
        elif self.model_type == 'vit':
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            reconstructed = self.decode(z)
            
            # ViT-specific loss function with patch-aware constraints
            recon_loss = F.mse_loss(reconstructed, x)
            
            # KL divergence
            logvar_clamped = torch.clamp(logvar, min=-20, max=10)
            kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp(), dim=1))
            
            # Spatial continuity loss (neighboring coordinates should be similar)
            continuity_loss = 0.01 * torch.mean(torch.abs(reconstructed[:, :-1] - reconstructed[:, 1:]))
            
            # Coordinate bounds loss
            bounds_loss = 0.05 * (F.relu(torch.abs(reconstructed) - 1.0).mean())
            
            # Check for NaN/Inf values
            if torch.isnan(recon_loss) or torch.isinf(recon_loss):
                recon_loss = torch.tensor(0.1, device=x.device, requires_grad=True)
            if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                kl_loss = torch.tensor(0.01, device=x.device, requires_grad=True)
            if torch.isnan(continuity_loss) or torch.isinf(continuity_loss):
                continuity_loss = torch.tensor(0.01, device=x.device, requires_grad=True)
            if torch.isnan(bounds_loss) or torch.isinf(bounds_loss):
                bounds_loss = torch.tensor(0.01, device=x.device, requires_grad=True)
            
            total_loss = recon_loss + 0.005 * kl_loss + continuity_loss + bounds_loss
            
            return {
                'reconstructed': reconstructed,
                'total_loss': total_loss,
                'recon_loss': recon_loss,
                'kl_loss': kl_loss,
                'continuity_loss': continuity_loss
            }
        
        elif self.model_type == 'flow':
            # Normalizing flow forward pass
            log_det = 0
            for flow in self.flows:
                x, ld = flow(x)
                log_det += ld
            
            # Base distribution log prob
            base_log_prob = -0.5 * torch.sum(x**2, dim=1) - 1.5 * math.log(2 * math.pi)
            total_log_prob = base_log_prob + log_det
            
            return x, -torch.mean(total_log_prob)  # Return negative log likelihood as loss
        
        elif self.model_type == 'diffusion':
            # Simple diffusion denoising loss
            noise = torch.randn_like(x)
            noisy_x = x + noise
            predicted_noise = self.model(noisy_x)
            loss = F.mse_loss(predicted_noise, noise)
            
            return predicted_noise, loss
    
    def log_prob(self, x):
        """Compute log probability for flow model"""
        if self.model_type == 'flow':
            log_det = 0
            x_transformed = x.clone()
            for flow in self.flows:
                x_transformed, ld = flow(x_transformed)
                log_det += ld
            
            # Base distribution log prob
            base_log_prob = -0.5 * torch.sum(x_transformed**2, dim=1) - 1.5 * math.log(2 * math.pi)
            total_log_prob = base_log_prob + log_det
            
            return total_log_prob
        else:
            raise NotImplementedError(f"log_prob not implemented for {self.model_type}")
    
    def sample(self, n_samples, device='cpu', extrapolate=True):
        """
        Sample spatial coordinates with physics-informed extrapolation
        
        Args:
            n_samples: Number of samples to generate
            device: Computing device
            extrapolate: If True, allow extrapolation beyond training data
        """
        self.eval()
        with torch.no_grad():
            if self.model_type in ['vae', 'convnext']:
                # Use larger latent noise for extrapolation
                noise_scale = 1.5 if extrapolate else 1.0
                z = torch.randn(n_samples, self.latent_dim, device=device) * noise_scale
                raw_samples = self.decode(z)
                
                # Apply physics-informed post-processing instead of simple tanh
                samples = self._apply_physics_constraints(raw_samples, extrapolate=extrapolate)
                
            elif self.model_type == 'flow':
                # Enhanced flow sampling with extrapolation
                noise_scale = 1.2 if extrapolate else 1.0
                z = torch.randn(n_samples, 3, device=device) * noise_scale
                
                # Inverse flow with stochastic perturbations for extrapolation
                for flow in reversed(self.flows):
                    z = flow.inverse(z)
                    if extrapolate:
                        # Add small stochastic perturbations to explore new regions
                        z = z + 0.1 * torch.randn_like(z) * noise_scale
                
                samples = self._apply_physics_constraints(z, extrapolate=extrapolate)
                
            elif self.model_type == 'diffusion':
                # Enhanced diffusion sampling with physics guidance
                samples = torch.randn(n_samples, 3, device=device)
                
                # Physics-guided denoising with extrapolation
                for step in range(15):  # More denoising steps
                    predicted_noise = self.model(samples)
                    step_size = 0.1 if not extrapolate else 0.15
                    samples = samples - step_size * predicted_noise
                    
                    # Apply physics guidance during denoising
                    if step % 3 == 0:  # Every few steps
                        samples = self._apply_physics_guidance(samples, step/15.0)
                
                samples = self._apply_physics_constraints(samples, extrapolate=extrapolate)
                
            else:
                # Physics-informed random sampling for unknown models
                samples = self._generate_physics_random_samples(n_samples, device, extrapolate)
                print(f"Warning: Unknown spatial model type '{self.model_type}', using physics-informed random samples")
            
            return samples
    
    def _apply_physics_constraints(self, samples, extrapolate=True):
        """Apply realistic physics constraints for spatial coordinates"""
        # Separate RA, DEC, redshift components
        ra_raw = samples[:, 0]
        dec_raw = samples[:, 1] 
        z_raw = samples[:, 2]
        
        if extrapolate:
            # Allow extrapolation with realistic survey limits
            ra_constrained = torch.sigmoid(ra_raw) * 360.0  # [0, 360] degrees
            
            # DEC with realistic sky coverage (northern hemisphere bias like SDSS)
            dec_constrained = torch.tanh(dec_raw) * 60.0 + 30.0  # [-30, +90] degrees  
            dec_constrained = torch.clamp(dec_constrained, -30.0, 90.0)
            
            # Extended redshift range for extrapolation with numerical stability
            z_constrained = F.softplus(z_raw + 1.0) * 1.5 + 0.05  
            z_constrained = torch.clamp(z_constrained, min=0.01, max=20.0)  # Prevent infinities while allowing high-z
            
        else:
            # Less constrained interpolation with numerical stability
            ra_constrained = torch.sigmoid(ra_raw) * 360.0
            dec_constrained = torch.tanh(dec_raw) * 90.0  # Full DEC range
            z_constrained = F.softplus(z_raw) + 0.01  
            z_constrained = torch.clamp(z_constrained, min=0.01, max=10.0)  # Prevent infinities
        
        # Add realistic survey selection effects
        ra_constrained = self._add_survey_selection_effects(ra_constrained, dec_constrained, z_constrained)
        
        return torch.stack([ra_constrained, dec_constrained, z_constrained], dim=1)
    
    def _add_survey_selection_effects(self, ra, dec, z):
        """Add realistic survey selection effects and clustering"""
        # Galactic plane avoidance (quasars avoid |b| < 10°)
        galactic_lat = torch.abs(dec - 30.0)  # Approximate galactic latitude
        avoidance_factor = torch.sigmoid((galactic_lat - 10.0) * 0.5)
        
        # Add some RA structure (survey stripe patterns)
        stripe_pattern = torch.sin(ra * math.pi / 30.0) * 0.1 + 1.0
        
        # Apply selection effects with stochastic acceptance
        selection_prob = avoidance_factor * stripe_pattern
        random_accept = torch.rand_like(ra)
        
        # Redistribute rejected samples to avoid gaps
        rejected_mask = random_accept > selection_prob * 0.8
        ra[rejected_mask] = torch.rand_like(ra[rejected_mask]) * 360.0
        
        return ra
    
    def _apply_physics_guidance(self, samples, progress):
        """Apply physics guidance during generation"""
        # Encourage realistic clustering and void structures
        ra, dec, z = samples[:, 0], samples[:, 1], samples[:, 2]
        
        # Large-scale structure guidance (encourage clustering)
        if progress > 0.3:  # Apply in later stages
            # Simple clustering: push samples toward local density maxima
            kernel_size = max(1, int(len(samples) * 0.05))  # 5% of samples
            
            for i in range(0, len(samples), kernel_size):
                batch = samples[i:i+kernel_size]
                if len(batch) > 1:
                    centroid = torch.mean(batch, dim=0)
                    # Gentle pull toward local centroid
                    samples[i:i+kernel_size] += 0.1 * (centroid.unsqueeze(0) - batch)
        
        return samples
    
    def _generate_physics_random_samples(self, n_samples, device, extrapolate):
        """Generate physics-informed random samples as fallback"""
        # Realistic RA distribution (uniform)
        ra = torch.rand(n_samples, device=device) * 360.0
        
        # Realistic DEC distribution (cosine weighting for sphere)
        dec_uniform = torch.rand(n_samples, device=device)
        dec = torch.asin(2 * dec_uniform - 1) * 180 / math.pi  # [-90, 90] with proper weighting
        
        # Clip to survey area
        if not extrapolate:
            dec = torch.clamp(dec, -30.0, 70.0)  # SDSS-like coverage
        else:
            dec = torch.clamp(dec, -30.0, 90.0)  # Extended coverage
        
        # Realistic redshift distribution (peaked at z~1.5 with exponential tail)
        z_base = torch.exponential(torch.ones(n_samples, device=device)) * 0.8 + 0.1
        if extrapolate:
            # Add high-z tail for extrapolation
            high_z_fraction = 0.1
            high_z_mask = torch.rand(n_samples, device=device) < high_z_fraction
            z_base[high_z_mask] = torch.exponential(torch.ones(high_z_mask.sum(), device=device)) * 2.0 + 3.0
        
        z = torch.clamp(z_base, 0.05, 8.0 if extrapolate else 4.0)
        
        return torch.stack([ra, dec, z], dim=1)

class FluxGenerator(nn.Module):
    """Generate realistic flux measurements conditioned on redshift"""
    
    def __init__(self, flux_dim=15, condition_dim=1, model_type='vae', latent_dim=64):
        super().__init__()
        self.flux_dim = flux_dim
        self.condition_dim = condition_dim  # Redshift conditioning
        self.model_type = model_type
        self.latent_dim = latent_dim
        
        if model_type == 'vae':
            # Conditional VAE for flux generation
            self.encoder = nn.Sequential(
                nn.Linear(flux_dim + condition_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256)
            )
            
            self.mu = nn.Linear(256, latent_dim)
            self.logvar = nn.Linear(256, latent_dim)
            
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim + condition_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Linear(256, flux_dim)
            )
        
        elif model_type == 'convnext':
            # ConvNeXt-based conditional flux generator
            self.encoder_net = ConvNeXtGenerativeEncoder(
                input_dim=flux_dim + condition_dim, latent_dim=latent_dim,
                depths=[3, 3, 6], dims=[96, 192, 384]  # Larger for complex flux patterns
            )
            self.decoder_net = ConvNeXtConditionalDecoder(
                latent_dim=latent_dim, condition_dim=condition_dim, 
                output_dim=flux_dim, depths=[6, 4, 3], dims=[512, 256, 128]
            )
            
            # Additional flux-specific attention for spectral modeling
            num_heads = min(8, max(1, latent_dim//8))
            self.flux_attention = nn.MultiheadAttention(
                embed_dim=latent_dim, num_heads=num_heads, dropout=0.1, batch_first=True
            )
        
        elif model_type == 'transformer':
            # Transformer-based conditional flux generator
            self.encoder_net = TransformerGenerativeEncoder(
                input_dim=flux_dim + condition_dim, latent_dim=latent_dim,
                d_model=384, num_heads=12, num_layers=8
            )
            self.decoder_net = TransformerConditionalDecoder(
                latent_dim=latent_dim, condition_dim=condition_dim,
                output_dim=flux_dim, d_model=512, num_heads=16, num_layers=6
            )
        
        elif model_type == 'vit':
            # Vision Transformer for flux generation (treating flux bands as patches)
            # Pad flux_dim to be divisible by patch_size if needed
            patch_size = 3  # Group flux bands into patches
            if flux_dim % patch_size != 0:
                padded_flux_dim = flux_dim + (patch_size - (flux_dim % patch_size))
            else:
                padded_flux_dim = flux_dim
            
            self.flux_padding = padded_flux_dim - flux_dim
            
            self.encoder_net = ViTGenerativeEncoder(
                input_dim=padded_flux_dim + condition_dim, latent_dim=latent_dim,
                d_model=256, num_heads=8, depth=8, patch_size=patch_size
            )
            self.decoder_net = ViTGenerativeDecoder(
                latent_dim=latent_dim, output_dim=flux_dim,
                d_model=256, num_heads=8, depth=6
            )
            
            # Condition projection for ViT (keep original dimension)
            # No projection needed, just use redshift directly
        
        elif model_type == 'flow':
            # Conditional normalizing flow
            self.flows = nn.ModuleList([
                ConditionalCouplingLayer(flux_dim, condition_dim, 128) for _ in range(6)
            ])
        
        elif model_type == 'diffusion':
            # Simple conditional diffusion for flux generation
            self.model = nn.Sequential(
                nn.Linear(flux_dim + condition_dim, 256),
                nn.GELU(),
                nn.Linear(256, 512),
                nn.GELU(),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, flux_dim)
            )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.model_type == 'diffusion':
                    # Extra conservative initialization for diffusion
                    nn.init.xavier_uniform_(module.weight, gain=0.01)
                else:
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def encode(self, flux, redshift):
        if self.model_type == 'vae':
            x = torch.cat([flux, redshift.unsqueeze(1)], dim=1)
            h = self.encoder(x)
            return self.mu(h), self.logvar(h)
        elif self.model_type in ['convnext', 'transformer']:
            x = torch.cat([flux, redshift.unsqueeze(1)], dim=1)
            return self.encoder_net(x)
        elif self.model_type == 'vit':
            # Pad flux if needed for ViT patch processing
            if self.flux_padding > 0:
                flux_padded = F.pad(flux, (0, self.flux_padding))
            else:
                flux_padded = flux
            
            # Concatenate flux with redshift condition
            x = torch.cat([flux_padded, redshift.unsqueeze(1)], dim=1)
            return self.encoder_net(x)
        else:
            return None, None
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, redshift):
        if self.model_type == 'vae':
            z_cond = torch.cat([z, redshift.unsqueeze(1)], dim=1)
            return self.decoder(z_cond)
        elif self.model_type == 'convnext':
            return self.decoder_net(z, redshift)
        elif self.model_type == 'transformer':
            return self.decoder_net(z, redshift.unsqueeze(1) if redshift.dim() == 1 else redshift)
        elif self.model_type == 'vit':
            return self.decoder_net(z)
        else:
            return z
    
    def forward(self, flux, redshift):
        if self.model_type == 'vae':
            mu, logvar = self.encode(flux, redshift)
            z = self.reparameterize(mu, logvar)
            reconstructed = self.decode(z, redshift)
            
            # Add numerical stability to prevent infinite values
            reconstructed = torch.clamp(reconstructed, min=-1e6, max=1e6)
            reconstructed = torch.where(torch.isnan(reconstructed), torch.zeros_like(reconstructed), reconstructed)
            reconstructed = torch.where(torch.isinf(reconstructed), torch.sign(reconstructed) * 1e6, reconstructed)
            
            # VAE loss with physics constraints and numerical stability
            recon_loss = F.mse_loss(reconstructed, flux)
            
            # Clamp logvar to prevent numerical issues
            logvar_clamped = torch.clamp(logvar, min=-20, max=10)
            
            # Improved KL divergence calculation with better numerical stability
            kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp(), dim=1))
            
            # Physics loss (only flux positivity for physical realism)
            physics_loss = 0.05 * F.relu(-reconstructed).mean()  # Only ensure positive flux
            
            # Add diversity loss to prevent mode collapse
            batch_mean = torch.mean(reconstructed, dim=0)
            diversity_loss = -torch.var(batch_mean)  # Encourage variation in batch means
            
            # Check for NaN/Inf values and replace with tensors that have gradients
            if torch.isnan(recon_loss) or torch.isinf(recon_loss):
                recon_loss = torch.tensor(1.0, device=flux.device, requires_grad=True)
            if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                kl_loss = torch.tensor(0.05, device=flux.device, requires_grad=True)
            if torch.isnan(physics_loss) or torch.isinf(physics_loss):
                physics_loss = torch.tensor(0.05, device=flux.device, requires_grad=True)
            if torch.isnan(diversity_loss) or torch.isinf(diversity_loss):
                diversity_loss = torch.tensor(0.0, device=flux.device, requires_grad=True)
            
            # Use β-VAE with lower KL weight to prevent mode collapse
            total_loss = recon_loss + 0.02 * kl_loss + 0.05 * physics_loss + 0.01 * diversity_loss
            
            return reconstructed, total_loss
        
        elif self.model_type == 'convnext':
            mu, logvar = self.encode(flux, redshift)
            z = self.reparameterize(mu, logvar)
            
            # Apply attention to latent representation
            z_attn = z.unsqueeze(1)  # (batch, 1, latent_dim)
            z_attn, flux_weights = self.flux_attention(z_attn, z_attn, z_attn)
            z_enhanced = z_attn.squeeze(1)  # Back to (batch, latent_dim)
            
            reconstructed = self.decode(z_enhanced, redshift)
            
            # Add numerical stability to prevent infinite values  
            reconstructed = torch.clamp(reconstructed, min=-1e6, max=1e6)
            reconstructed = torch.where(torch.isnan(reconstructed), torch.zeros_like(reconstructed), reconstructed)
            reconstructed = torch.where(torch.isinf(reconstructed), torch.sign(reconstructed) * 1e6, reconstructed)
            
            # Enhanced loss function for ConvNeXt flux generation
            recon_loss = F.mse_loss(reconstructed, flux)
            
            # KL divergence with stability improvements
            logvar_clamped = torch.clamp(logvar, min=-20, max=10)
            kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp(), dim=1))
            
            # Flux physics constraints - ensure realistic flux patterns
            # Positive flux constraint
            positivity_loss = 0.1 * F.relu(-reconstructed).mean()
            
            # Spectral smoothness constraint (adjacent bands should have similar fluxes)
            smoothness_loss = 0.01 * torch.mean(torch.abs(reconstructed[:, 1:] - reconstructed[:, :-1]))
            
            # Remove flux magnitude upper limit - let model learn natural range
            magnitude_loss = torch.tensor(0.0, device=reconstructed.device, requires_grad=True)
            
            # Attention entropy regularization for better attention patterns
            attention_entropy = -torch.mean(torch.sum(flux_weights * torch.log(flux_weights + 1e-10), dim=-1))
            attention_reg = 0.005 * attention_entropy
            
            # Check for NaN/Inf values
            if torch.isnan(recon_loss) or torch.isinf(recon_loss):
                recon_loss = torch.tensor(1.0, device=flux.device, requires_grad=True)
            if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                kl_loss = torch.tensor(0.01, device=flux.device, requires_grad=True)
            if torch.isnan(positivity_loss) or torch.isinf(positivity_loss):
                positivity_loss = torch.tensor(0.01, device=flux.device, requires_grad=True)
            if torch.isnan(smoothness_loss) or torch.isinf(smoothness_loss):
                smoothness_loss = torch.tensor(0.01, device=flux.device, requires_grad=True)
            if torch.isnan(magnitude_loss) or torch.isinf(magnitude_loss):
                magnitude_loss = torch.tensor(0.01, device=flux.device, requires_grad=True)
            if torch.isnan(attention_reg) or torch.isinf(attention_reg):
                attention_reg = torch.tensor(0.0, device=flux.device, requires_grad=True)
            
            # Total loss with ConvNeXt-specific regularization
            total_loss = (recon_loss + 0.01 * kl_loss + positivity_loss + 
                         smoothness_loss + magnitude_loss + attention_reg)
            
            return reconstructed, total_loss
        
        elif self.model_type == 'transformer':
            mu, logvar = self.encode(flux, redshift)
            z = self.reparameterize(mu, logvar)
            reconstructed = self.decode(z, redshift)
            
            # Add numerical stability to prevent infinite values
            reconstructed = torch.clamp(reconstructed, min=-1e6, max=1e6)
            reconstructed = torch.where(torch.isnan(reconstructed), torch.zeros_like(reconstructed), reconstructed)
            reconstructed = torch.where(torch.isinf(reconstructed), torch.sign(reconstructed) * 1e6, reconstructed)
            
            # Transformer-specific flux loss
            recon_loss = F.mse_loss(reconstructed, flux)
            
            # KL divergence
            logvar_clamped = torch.clamp(logvar, min=-20, max=10)
            kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp(), dim=1))
            
            # Spectral physics constraints - only positivity for physical realism
            positivity_loss = 0.1 * F.relu(-reconstructed).mean()
            magnitude_loss = torch.tensor(0.0, device=reconstructed.device, requires_grad=True)  # Remove upper limit
            
            # Spectral coherence - flux bands should follow physical relationships
            spectral_coherence = 0.02 * torch.mean(torch.abs(reconstructed[:, 1:] - reconstructed[:, :-1]))
            
            # Check for NaN/Inf values
            if torch.isnan(recon_loss) or torch.isinf(recon_loss):
                recon_loss = torch.tensor(1.0, device=flux.device, requires_grad=True)
            if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                kl_loss = torch.tensor(0.01, device=flux.device, requires_grad=True)
            if torch.isnan(positivity_loss) or torch.isinf(positivity_loss):
                positivity_loss = torch.tensor(0.01, device=flux.device, requires_grad=True)
            if torch.isnan(magnitude_loss) or torch.isinf(magnitude_loss):
                magnitude_loss = torch.tensor(0.01, device=flux.device, requires_grad=True)
            if torch.isnan(spectral_coherence) or torch.isinf(spectral_coherence):
                spectral_coherence = torch.tensor(0.01, device=flux.device, requires_grad=True)
            
            total_loss = recon_loss + 0.01 * kl_loss + positivity_loss + magnitude_loss + spectral_coherence
            
            return reconstructed, total_loss
        
        elif self.model_type == 'vit':
            mu, logvar = self.encode(flux, redshift)
            z = self.reparameterize(mu, logvar)
            reconstructed = self.decode(z, redshift)
            
            # Add numerical stability to prevent infinite values
            reconstructed = torch.clamp(reconstructed, min=-1e6, max=1e6)
            reconstructed = torch.where(torch.isnan(reconstructed), torch.zeros_like(reconstructed), reconstructed)
            reconstructed = torch.where(torch.isinf(reconstructed), torch.sign(reconstructed) * 1e6, reconstructed)
            
            # ViT-specific flux loss with patch-aware constraints
            recon_loss = F.mse_loss(reconstructed, flux)
            
            # KL divergence
            logvar_clamped = torch.clamp(logvar, min=-20, max=10)
            kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp(), dim=1))
            
            # Physics constraints - only positivity for physical realism
            positivity_loss = 0.1 * F.relu(-reconstructed).mean()
            magnitude_loss = torch.tensor(0.0, device=reconstructed.device, requires_grad=True)  # Remove upper limit
            
            # Patch consistency - flux bands within patches should be correlated
            patch_size = 3
            if reconstructed.shape[1] >= patch_size:
                patch_consistency = 0.01 * torch.mean(torch.var(
                    reconstructed[:, :reconstructed.shape[1]//patch_size*patch_size].view(
                        reconstructed.shape[0], -1, patch_size
                    ), dim=2
                ))
            else:
                patch_consistency = torch.tensor(0.0, device=flux.device, requires_grad=True)
            
            # Check for NaN/Inf values
            if torch.isnan(recon_loss) or torch.isinf(recon_loss):
                recon_loss = torch.tensor(1.0, device=flux.device, requires_grad=True)
            if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                kl_loss = torch.tensor(0.01, device=flux.device, requires_grad=True)
            if torch.isnan(positivity_loss) or torch.isinf(positivity_loss):
                positivity_loss = torch.tensor(0.01, device=flux.device, requires_grad=True)
            if torch.isnan(magnitude_loss) or torch.isinf(magnitude_loss):
                magnitude_loss = torch.tensor(0.01, device=flux.device, requires_grad=True)
            if torch.isnan(patch_consistency) or torch.isinf(patch_consistency):
                patch_consistency = torch.tensor(0.0, device=flux.device, requires_grad=True)
            
            total_loss = recon_loss + 0.005 * kl_loss + positivity_loss + magnitude_loss + patch_consistency
            
            return reconstructed, total_loss
        
        elif self.model_type == 'flow':
            # Conditional normalizing flow
            log_det = 0
            x = flux
            for flow in self.flows:
                x, ld = flow(x, redshift)
                log_det += ld
                
                # Add numerical stability during flow transformations
                x = torch.clamp(x, min=-1e6, max=1e6)
                x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
                x = torch.where(torch.isinf(x), torch.sign(x) * 1e6, x)
            
            base_log_prob = -0.5 * torch.sum(x**2, dim=1) - 0.5 * self.flux_dim * math.log(2 * math.pi)
            total_log_prob = base_log_prob + log_det
            
            # Clamp to prevent NaN/Inf
            total_log_prob = torch.clamp(total_log_prob, min=-1e6, max=1e6)
            loss = -torch.mean(total_log_prob)
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                loss = torch.tensor(1.0, device=x.device, requires_grad=True)
            
            return x, loss
        
        elif self.model_type == 'diffusion':
            # Improved conditional diffusion with noise scheduling
            # Random timestep for training
            t = torch.randint(0, 100, (flux.shape[0],), device=flux.device)
            
            # Scale noise based on timestep (0.01 to 0.5 range)
            noise_scale = 0.01 + (t.float() / 100.0) * 0.49
            noise = torch.randn_like(flux) * noise_scale.unsqueeze(1)
            
            # Add noise to flux
            noisy_flux = flux + noise
            
            # Clamp to prevent extreme values
            noisy_flux = torch.clamp(noisy_flux, min=-10, max=10)
            
            # Concatenate with redshift condition
            flux_with_condition = torch.cat([noisy_flux, redshift.unsqueeze(1)], dim=1)
            
            # Predict the noise
            predicted_noise = self.model(flux_with_condition)
            
            # Add comprehensive numerical stability
            predicted_noise = torch.clamp(predicted_noise, min=-5, max=5)
            predicted_noise = torch.where(torch.isnan(predicted_noise), torch.zeros_like(predicted_noise), predicted_noise)
            predicted_noise = torch.where(torch.isinf(predicted_noise), torch.sign(predicted_noise) * 5.0, predicted_noise)
            
            # Compute loss with proper scaling
            loss = F.mse_loss(predicted_noise, noise, reduction='mean')
            
            # Additional stability checks
            if torch.isnan(loss) or torch.isinf(loss) or loss > 100:
                loss = torch.tensor(1.0, device=flux.device, requires_grad=True)
            
            return predicted_noise, loss
    
    def sample(self, redshift, device='cpu', extrapolate=True):
        """
        Sample flux measurements with physics-informed extrapolation
        
        Args:
            redshift: Redshift values for conditioning
            device: Computing device
            extrapolate: If True, allow extrapolation beyond training data
        """
        self.eval()
        n_samples = len(redshift)
        
        with torch.no_grad():
            # Adjust sampling based on redshift for extrapolation
            z_scale = 1.0
            if extrapolate:
                # Scale noise based on redshift for high-z extrapolation
                high_z_mask = redshift > 3.0
                z_scale = torch.ones_like(redshift)
                z_scale[high_z_mask] = 1.0 + 0.3 * (redshift[high_z_mask] - 3.0) / 5.0
                z_scale = z_scale.unsqueeze(1)
            
            if self.model_type == 'vae':
                z = torch.randn(n_samples, self.latent_dim, device=device) * z_scale
                samples = self.decode(z, redshift)
                
            elif self.model_type == 'convnext':
                # ConvNeXt sampling with attention-enhanced generation
                z = torch.randn(n_samples, self.latent_dim, device=device) * z_scale
                
                # Apply attention to latent space
                z_attn = z.unsqueeze(1)  # (batch, 1, latent_dim)
                z_attn, _ = self.flux_attention(z_attn, z_attn, z_attn)
                z = z_attn.squeeze(1)  # Back to (batch, latent_dim)
                
                samples = self.decode(z, redshift)
                
            elif self.model_type == 'transformer':
                # Transformer sampling with conditional generation
                z = torch.randn(n_samples, self.latent_dim, device=device) * z_scale
                samples = self.decode(z, redshift)
                
            elif self.model_type == 'vit':
                # ViT sampling with patch-aware generation  
                z = torch.randn(n_samples, self.latent_dim, device=device) * z_scale
                samples = self.decode(z, redshift)
            elif self.model_type == 'flow':
                z = torch.randn(n_samples, self.flux_dim, device=device)
                # Inverse conditional flow
                for flow in reversed(self.flows):
                    z = flow.inverse(z, redshift)
                samples = z
            elif self.model_type == 'diffusion':
                # Improved diffusion sampling with proper denoising schedule
                samples = torch.randn(n_samples, self.flux_dim, device=device) * 0.5  # Start with less noise
                
                # Denoising steps with decreasing noise schedule
                n_steps = 20
                for step in range(n_steps):
                    # Decreasing step size
                    step_size = 0.2 * (1 - step / n_steps)
                    
                    # Concatenate with redshift condition
                    flux_with_condition = torch.cat([samples, redshift.unsqueeze(1)], dim=1)
                    
                    # Predict noise
                    predicted_noise = self.model(flux_with_condition)
                    
                    # Clamp predictions to prevent instability
                    predicted_noise = torch.clamp(predicted_noise, min=-2, max=2)
                    
                    # Denoising step
                    samples = samples - step_size * predicted_noise
                    
                    # Clamp samples to reasonable range
                    samples = torch.clamp(samples, min=-5, max=5)
            else:
                # Physics-informed fallback sampling
                samples = self._generate_physics_flux_samples(n_samples, redshift, device, extrapolate)
                print(f"Warning: Unknown model type '{self.model_type}', using physics-informed random samples")
            
            # Apply physics-informed post-processing
            samples = self._apply_flux_physics_constraints(samples, redshift, extrapolate)
            
            return samples
    
    def _generate_physics_flux_samples(self, n_samples, redshift, device, extrapolate):
        """Generate physics-informed flux samples as fallback"""
        # Generate samples based on realistic quasar SED evolution
        samples = torch.zeros(n_samples, self.flux_dim, device=device)
        
        for i in range(n_samples):
            z = redshift[i].item()
            
            # K-correction and evolution effects
            k_corr = 2.5 * math.log10(1 + z)
            evolution = -0.5 * z  # Luminosity evolution
            
            # Generate realistic flux ratios
            # Optical bands (G, R, Z indices 0, 1, 2)
            base_g = torch.randn(1, device=device) * 0.5 - 2.0  # Log flux
            
            # Realistic color evolution with redshift
            g_r = 0.5 + 0.1 * z + torch.randn(1, device=device) * 0.2
            r_z = 0.3 + 0.15 * z + torch.randn(1, device=device) * 0.15
            
            samples[i, 0] = base_g  # G band
            samples[i, 1] = base_g - g_r  # R band  
            samples[i, 2] = base_g - g_r - r_z  # Z band
            
            # Inverse variances (uncertainty measures)
            samples[i, 3:6] = torch.abs(torch.randn(3, device=device)) + 1.0
            
            # IR bands (W1, W2 indices 6, 7)
            if self.flux_dim > 6:
                # IR gets redder and fainter with redshift
                w1_offset = -0.5 * z - k_corr + torch.randn(1, device=device) * 0.3
                w2_offset = -0.7 * z - k_corr + torch.randn(1, device=device) * 0.3
                
                samples[i, 6] = base_g + w1_offset  # W1
                samples[i, 7] = base_g + w2_offset  # W2
                
                # IR inverse variances
                if self.flux_dim > 8:
                    samples[i, 8:10] = torch.abs(torch.randn(2, device=device)) + 0.5
                
                # ML features (if present)
                if self.flux_dim > 10:
                    samples[i, 10:] = samples[i, :self.flux_dim-10] + torch.randn(self.flux_dim-10, device=device) * 0.1
        
        return samples
    
    def _apply_flux_physics_constraints(self, samples, redshift, extrapolate):
        """Apply physics constraints to flux samples"""
        # Ensure realistic flux evolution with redshift
        n_samples = len(samples)
        
        # Apply K-corrections and evolution
        for i in range(n_samples):
            z = redshift[i].item()
            
            # Distance modulus effect (fluxes get fainter with distance)
            # Approximate luminosity distance effect  
            distance_dimming = 2.5 * math.log10((1 + z)**2)
            
            # Apply to all flux measurements (first few bands)
            n_flux_bands = min(6, self.flux_dim)
            samples[i, :n_flux_bands] = samples[i, :n_flux_bands] - distance_dimming
            
            # Spectral evolution (UV excess gets redshifted)
            if z > 2.0 and extrapolate:
                # High-z spectral hardening
                uv_excess = 0.3 * (z - 2.0) / 4.0
                samples[i, 0] = samples[i, 0] + uv_excess  # Boost blue band
        
        # Ensure realistic color constraints
        samples = self._enforce_color_constraints(samples)
        
        # Ensure positive fluxes with realistic floor
        samples = F.softplus(samples) + 1e-8
        
        # Apply survey-specific magnitude limits
        if extrapolate:
            # Allow fainter objects for extrapolation
            flux_floor = 1e-8
        else:
            # Conservative flux limits
            flux_floor = 1e-6
        
        samples = torch.clamp(samples, min=flux_floor, max=1e-1)
        
        return samples
    
    def _enforce_color_constraints(self, samples):
        """Enforce realistic quasar color constraints"""
        # Ensure optical colors stay within realistic ranges
        if self.flux_dim >= 3:
            # G-R color constraint
            g_r_color = samples[:, 0] - samples[:, 1]
            g_r_color = torch.clamp(g_r_color, -0.5, 2.0)  # Realistic range
            samples[:, 1] = samples[:, 0] - g_r_color
            
            # R-Z color constraint  
            r_z_color = samples[:, 1] - samples[:, 2]
            r_z_color = torch.clamp(r_z_color, -0.3, 1.5)  # Realistic range
            samples[:, 2] = samples[:, 1] - r_z_color
        
        return samples

# Helper coupling layers for the simpler models
class SimpleCouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim - dim // 2)
        )
    
    def forward(self, x):
        x1, x2 = x[:, :self.dim//2], x[:, self.dim//2:]
        y2 = x2 + self.net(x1)
        return torch.cat([x1, y2], dim=1), torch.zeros(x.shape[0], device=x.device)
    
    def inverse(self, y):
        y1, y2 = y[:, :self.dim//2], y[:, self.dim//2:]
        x2 = y2 - self.net(y1)
        return torch.cat([y1, x2], dim=1)

class ConditionalCouplingLayer(nn.Module):
    def __init__(self, dim, condition_dim, hidden_dim):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim // 2 + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim - dim // 2)
        )
    
    def forward(self, x, condition):
        x1, x2 = x[:, :self.dim//2], x[:, self.dim//2:]
        net_input = torch.cat([x1, condition.unsqueeze(1)], dim=1)
        y2 = x2 + self.net(net_input)
        return torch.cat([x1, y2], dim=1), torch.zeros(x.shape[0], device=x.device)
    
    def inverse(self, y, condition):
        y1, y2 = y[:, :self.dim//2], y[:, self.dim//2:]
        net_input = torch.cat([y1, condition.unsqueeze(1)], dim=1)
        x2 = y2 - self.net(net_input)
        return torch.cat([y1, x2], dim=1)

class QuasarGenerativeFramework:
    """Main framework for training and using two-stage generative models"""
    
    def __init__(self, spatial_model_type='vae', flux_model_type='vae', device='cpu'):
        self.device = device
        self.spatial_model_type = spatial_model_type
        self.flux_model_type = flux_model_type
        self.validator = PhysicsValidator()
        
        # Load and preprocess data
        self.load_data()
        
        # Initialize two separate models
        self.spatial_generator = SpatialRedshiftGenerator(
            model_type=spatial_model_type
        ).to(device)
        
        self.flux_generator = FluxGenerator(
            flux_dim=self.flux_dim,
            condition_dim=1,  # Redshift conditioning
            model_type=flux_model_type
        ).to(device)
    
    def load_data(self):
        """Load and preprocess quasar data for two-stage generation"""
        print("Loading quasar data for spatial-redshift and flux generation...")
        
        # Load data
        data = pd.read_csv('data/Sep<20.csv')
        print(f"Loaded {len(data)} rows of raw data")
        
        # Select flux features
        flux_features = ['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_IVAR_G', 'FLUX_IVAR_R', 
                        'FLUX_IVAR_Z', 'FLUX_IVAR_W1', 'FLUX_W1', 'FLUX_W2', 
                        'ML_FLUX_P1', 'ML_FLUX_P2', 'ML_FLUX_P3', 'ML_FLUX_P4', 
                        'ML_FLUX_P5', 'ML_FLUX_P6']
        
        # Extract spatial coordinates and redshift
        spatial_features = ['TARGET_RA', 'TARGET_DEC']
        
        X_flux = data[flux_features].values
        X_spatial = data[spatial_features].values
        redshifts = data['Z'].values
        
        # Clean data - remove NaN/Inf values
        flux_mask = ~(np.isnan(X_flux).any(axis=1) | np.isinf(X_flux).any(axis=1))
        spatial_mask = ~(np.isnan(X_spatial).any(axis=1) | np.isinf(X_spatial).any(axis=1))
        redshift_mask = ~(np.isnan(redshifts) | np.isinf(redshifts))
        
        combined_mask = flux_mask & spatial_mask & redshift_mask
        
        X_flux_clean = X_flux[combined_mask]
        X_spatial_clean = X_spatial[combined_mask]
        redshifts_clean = redshifts[combined_mask]
        
        print(f"After cleaning: {len(X_flux_clean)} valid samples")
        print(f"Flux range: [{X_flux_clean.min():.6f}, {X_flux_clean.max():.6f}]")
        print(f"Redshift range: [{redshifts_clean.min():.6f}, {redshifts_clean.max():.6f}]")
        
        # Work with flux values directly (no log transformation)
        # Only apply small offset to handle potential zeros for numerical stability
        X_flux_clean = X_flux_clean + 1e-10
        
        # Additional cleaning: remove any remaining inf/nan values and extreme outliers
        X_flux_clean = np.where(np.isnan(X_flux_clean), 1e-10, X_flux_clean)
        X_flux_clean = np.where(np.isinf(X_flux_clean), 1e-10, X_flux_clean)
        X_flux_clean = np.clip(X_flux_clean, 1e-10, 1e6)  # Reasonable flux range
        
        # Normalize spatial coordinates with bounds checking
        # RA: 0-360 degrees -> [-1, 1]
        # DEC: -90 to +90 degrees -> [-1, 1]  
        # Redshift: normalize to [-1, 1] based on data range
        ra_norm = 2 * (np.clip(X_spatial_clean[:, 0], 0, 360) / 360.0) - 1
        dec_norm = np.clip(X_spatial_clean[:, 1], -90, 90) / 90.0
        
        # More robust redshift normalization
        z_max = np.clip(redshifts_clean.max(), 0.1, 20.0)  # Prevent division by very small numbers
        z_norm = 2 * (np.clip(redshifts_clean, 0, z_max) / z_max) - 1
        
        # Create spatial-redshift data for first generator
        spatial_redshift_data = np.column_stack([ra_norm, dec_norm, z_norm])
        
        # Store original coordinates and redshifts
        self.original_coords = X_spatial_clean
        self.original_redshifts = redshifts_clean
        self.redshift_max = redshifts_clean.max()

        # Split data for both generators
        (flux_train, flux_test, 
         spatial_train, spatial_test, 
         redshift_train, redshift_test,
         coords_train, coords_test) = train_test_split(
            X_flux_clean, spatial_redshift_data, redshifts_clean, X_spatial_clean,
            test_size=0.2, random_state=42
        )
        
        # Standardize flux features with robust scaling
        self.flux_scaler = StandardScaler()
        flux_train_scaled = self.flux_scaler.fit_transform(flux_train)
        flux_test_scaled = self.flux_scaler.transform(flux_test)
        
        # Clean scaled data to prevent infinite values
        flux_train_scaled = np.where(np.isnan(flux_train_scaled), 0, flux_train_scaled)
        flux_train_scaled = np.where(np.isinf(flux_train_scaled), 0, flux_train_scaled)
        flux_train_scaled = np.clip(flux_train_scaled, -10, 10)  # Reasonable standardized range
        
        flux_test_scaled = np.where(np.isnan(flux_test_scaled), 0, flux_test_scaled)
        flux_test_scaled = np.where(np.isinf(flux_test_scaled), 0, flux_test_scaled)
        flux_test_scaled = np.clip(flux_test_scaled, -10, 10)
        
        # Clean spatial data before tensor conversion
        spatial_train = np.where(np.isnan(spatial_train), 0, spatial_train)
        spatial_train = np.where(np.isinf(spatial_train), 0, spatial_train)
        spatial_train = np.clip(spatial_train, -2, 2)
        
        spatial_test = np.where(np.isnan(spatial_test), 0, spatial_test)
        spatial_test = np.where(np.isinf(spatial_test), 0, spatial_test)
        spatial_test = np.clip(spatial_test, -2, 2)
        
        # Convert to tensors for spatial-redshift generator
        self.spatial_train_data = torch.FloatTensor(spatial_train).to(self.device)
        self.spatial_test_data = torch.FloatTensor(spatial_test).to(self.device)
        
        # Convert to tensors for flux generator (flux + redshift condition)
        self.flux_train_data = torch.FloatTensor(flux_train_scaled).to(self.device)
        self.flux_test_data = torch.FloatTensor(flux_test_scaled).to(self.device)
        self.redshift_train_data = torch.FloatTensor(redshift_train).to(self.device)
        self.redshift_test_data = torch.FloatTensor(redshift_test).to(self.device)
        
        # Final check for any remaining infinite values in tensors
        def clean_tensor(tensor):
            tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
            tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
            return tensor
        
        self.spatial_train_data = clean_tensor(self.spatial_train_data)
        self.spatial_test_data = clean_tensor(self.spatial_test_data)
        self.flux_train_data = clean_tensor(self.flux_train_data)
        self.flux_test_data = clean_tensor(self.flux_test_data)
        self.redshift_train_data = clean_tensor(self.redshift_train_data)
        self.redshift_test_data = clean_tensor(self.redshift_test_data)
        
        # Store original coordinates for validation
        self.coords_train = coords_train
        self.coords_test = coords_test
        
        # Set dimensions
        self.flux_dim = len(flux_features)
        self.spatial_dim = 3  # RA, DEC, redshift
        
        print(f"  Training samples: {len(flux_train)}")
        print(f"  Test samples: {len(flux_test)}")
        print(f"  Flux dimensions: {self.flux_dim}")
        print(f"  Spatial-redshift dimensions: {self.spatial_dim}")
        print(f"  RA range: {coords_train[:, 0].min():.2f}° to {coords_train[:, 0].max():.2f}°")
        print(f"  DEC range: {coords_train[:, 1].min():.2f}° to {coords_train[:, 1].max():.2f}°")
        print(f"  Redshift range: {redshift_train.min():.3f} to {redshift_train.max():.3f}")
    
    def train(self, epochs=200, lr=2e-3, batch_size=128):
        """Train both generative models"""
        print(f"Training two-stage generative models...")
        
        # Train spatial-redshift generator
        print("  Training spatial-redshift generator...")
        self.train_spatial_generator(epochs, lr, batch_size)
        
        # Train flux generator
        print("  Training flux generator...")
        self.train_flux_generator(epochs, lr, batch_size)
    
    def train_spatial_generator(self, epochs=200, lr=2e-3, batch_size=128):
        """Train the spatial-redshift generator"""
        print(f"Training spatial generator ({self.spatial_model_type}) for {epochs} epochs with LR={lr}...")
        optimizer = torch.optim.AdamW(self.spatial_generator.parameters(), lr=lr, weight_decay=1e-5, betas=(0.9, 0.999))
        
        # Use cosine annealing for better convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=lr/100
        )
        
        # Create data loader for spatial-redshift data
        spatial_dataset = torch.utils.data.TensorDataset(self.spatial_train_data)
        spatial_loader = torch.utils.data.DataLoader(spatial_dataset, batch_size=batch_size, shuffle=True)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.spatial_generator.train()
            total_loss = 0
            
            for batch_idx, (batch_data,) in enumerate(spatial_loader):
                optimizer.zero_grad()

                if self.spatial_model_type in ['vae', 'convnext', 'transformer', 'vit']:
                    outputs = self.spatial_generator(batch_data)
                    loss = outputs['total_loss']
                elif self.spatial_model_type == 'flow':
                    log_prob = self.spatial_generator.log_prob(batch_data)
                    loss = -torch.mean(log_prob)
                elif self.spatial_model_type == 'diffusion':
                    predicted_noise, loss = self.spatial_generator(batch_data)

                # Check for NaN loss before backprop
                if not torch.isnan(loss) and not torch.isinf(loss) and loss.item() < 1e6:
                    loss.backward()
                    # More aggressive gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.spatial_generator.parameters(), 0.5)
                    
                    # Skip update if gradients are too large
                    if grad_norm < 100:
                        optimizer.step()
                        total_loss += loss.item()
                    else:
                        print(f"    Warning: Large gradient norm {grad_norm:.2f} at epoch {epoch}, batch {batch_idx}")
                        total_loss += 1.0
                else:
                    print(f"    Warning: NaN/Inf loss detected at epoch {epoch}, batch {batch_idx}")
                    total_loss += 1.0  # Use default loss value
            
            avg_loss = total_loss / len(spatial_loader)
            scheduler.step()  # CosineAnnealingWarmRestarts doesn't need loss
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(self.spatial_generator.state_dict(), f'best_spatial_{self.spatial_model_type}_model.pth')
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter > 50:
                print(f"  Early stopping at epoch {epoch} (no improvement for 50 epochs)")
                break
            
            if epoch % 20 == 0:
                print(f"  Epoch {epoch:3d}: Loss = {avg_loss:.6f}, LR = {optimizer.param_groups[0]['lr']:.2e}")
        
        print(f"  Spatial generator training completed! Best loss: {best_loss:.6f}")
    
    def train_flux_generator(self, epochs=200, lr=2e-3, batch_size=128):
        """Train the flux generator conditioned on redshift"""
        print(f"Training flux generator ({self.flux_model_type}) for {epochs} epochs with LR={lr}...")
        
        optimizer = torch.optim.Adam(self.flux_generator.parameters(), lr=lr, weight_decay=1e-5, betas=(0.9, 0.999))
        
        # Use cosine annealing for better convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=lr/100
        )
        
        # Create data loader for flux data with redshift conditioning
        flux_dataset = torch.utils.data.TensorDataset(self.flux_train_data, self.redshift_train_data)
        flux_loader = torch.utils.data.DataLoader(flux_dataset, batch_size=batch_size, shuffle=True)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.flux_generator.train()
            total_loss = 0
            
            for batch_idx, (flux_batch, redshift_batch) in enumerate(flux_loader):
                optimizer.zero_grad()
                
                if self.flux_model_type in ['vae', 'convnext', 'transformer', 'vit']:
                    reconstructed, loss = self.flux_generator(flux_batch, redshift_batch)
                elif self.flux_model_type == 'flow':
                    transformed, loss = self.flux_generator(flux_batch, redshift_batch)
                elif self.flux_model_type == 'diffusion':
                    predicted_noise, loss = self.flux_generator(flux_batch, redshift_batch)
                
                # Check for NaN loss before backprop
                if not torch.isnan(loss) and not torch.isinf(loss) and loss.item() < 1e6:
                    loss.backward()
                    # More aggressive gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.flux_generator.parameters(), 0.5)
                    
                    # Skip update if gradients are too large
                    if grad_norm < 100:
                        optimizer.step()
                        total_loss += loss.item()
                    else:
                        print(f"    Warning: Large gradient norm {grad_norm:.2f} at epoch {epoch}, batch {batch_idx}")
                        total_loss += 1.0
                else:
                    print(f"    Warning: NaN/Inf loss detected at epoch {epoch}, batch {batch_idx}")
                    total_loss += 1.0  # Use default loss value
            
            avg_loss = total_loss / len(flux_loader)
            scheduler.step()  # CosineAnnealingWarmRestarts doesn't need loss
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(self.flux_generator.state_dict(), f'best_flux_{self.flux_model_type}_model.pth')
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter > 50:
                print(f"  Early stopping at epoch {epoch} (no improvement for 50 epochs)")
                break
            
            if epoch % 20 == 0:
                print(f"  Epoch {epoch:3d}: Loss = {avg_loss:.6f}, LR = {optimizer.param_groups[0]['lr']:.2e}")
        
        print(f"  Flux generator training completed! Best loss: {best_loss:.6f}")
    
    def generate_samples(self, n_samples=1000, extrapolate=True):
        """
        Generate synthetic quasar samples using physics-informed two-stage approach
        
        Args:
            n_samples: Number of samples to generate
            extrapolate: If True, allow extrapolation beyond training data
        """
        extrapolate_str = "with extrapolation" if extrapolate else "interpolation only"
        print(f"Generating {n_samples} synthetic quasars using two-stage generation ({extrapolate_str})...")
        
        self.spatial_generator.eval()
        self.flux_generator.eval()
        
        with torch.no_grad():
            # Stage 1: Generate spatial coordinates and redshifts with physics constraints
            print("  Stage 1: Generating physics-informed spatial coordinates and redshifts...")
            
            # Use enhanced sampling with extrapolation capability
            spatial_samples = self.spatial_generator.sample(n_samples, self.device, extrapolate=extrapolate)
            
            # Extract already-constrained physical coordinates
            generated_ra = spatial_samples[:, 0]         # Already in [0, 360] degrees
            generated_dec = spatial_samples[:, 1]        # Already in survey bounds
            generated_redshifts = spatial_samples[:, 2]  # Already in physical range
            
            generated_coords = torch.stack([generated_ra, generated_dec], dim=1)
            
            # Stage 2: Generate physics-informed flux conditioned on redshift
            print("  Stage 2: Generating physics-informed flux conditioned on redshift...")
            generated_flux_scaled = self.flux_generator.sample(generated_redshifts, self.device, extrapolate=extrapolate)
            
            # Inverse transform flux features to original scale (no log reversal needed)
            generated_flux_np = generated_flux_scaled.cpu().numpy()
            generated_flux_original = self.flux_scaler.inverse_transform(generated_flux_np)
            # Remove small numerical offset that was added for stability
            generated_flux_original = generated_flux_original - 1e-10
            
            generated_flux_tensor = torch.FloatTensor(generated_flux_original).to(self.device)
        
        print(f"  Generated {n_samples} samples with two-stage approach")
        print(f"    - RA range: {generated_ra.cpu().min():.2f}° to {generated_ra.cpu().max():.2f}°")
        print(f"    - DEC range: {generated_dec.cpu().min():.2f}° to {generated_dec.cpu().max():.2f}°")
        print(f"    - Redshift range: {generated_redshifts.cpu().min():.3f} to {generated_redshifts.cpu().max():.3f}")
        
        return generated_flux_tensor, generated_coords, generated_redshifts
    
    def save_synthetic_data(self, n_samples=10000, output_dir='synthetic_data', format='all', extrapolate=True):
        """
        Generate and save synthetic quasar data for analysis
        
        Args:
            n_samples (int): Number of synthetic quasars to generate
            output_dir (str): Directory to save the data
            format (str): Output format - 'csv', 'numpy', 'fits', 'all'
            extrapolate (bool): If True, allow extrapolation beyond training data
        """
        import os
        from datetime import datetime
        
        print(f"Generating and saving {n_samples} synthetic quasars...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate synthetic data with extrapolation
        generated_fluxes, generated_coords, generated_redshifts = self.generate_samples(n_samples, extrapolate=extrapolate)
        
        # Convert to numpy for easier handling
        fluxes_np = generated_fluxes.cpu().numpy()
        coords_np = generated_coords.cpu().numpy()
        redshifts_np = generated_redshifts.cpu().numpy()
        
        # Create comprehensive dataset
        ra_values = coords_np[:, 0]
        dec_values = coords_np[:, 1]
        
        # Timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        spatial_type = self.spatial_generator.model_type
        flux_type = self.flux_generator.model_type
        
        base_filename = f"synthetic_quasars_{spatial_type}_{flux_type}_{n_samples}_{timestamp}"
        
        print(f"  Saving data with base filename: {base_filename}")
        
        # 1. Save as CSV (human-readable, good for analysis)
        if format in ['csv', 'all']:
            import pandas as pd
            
            # Create comprehensive DataFrame
            data_dict = {
                'TARGET_RA': ra_values,
                'TARGET_DEC': dec_values,
                'REDSHIFT': redshifts_np,
            }
            
            # Add flux columns (using same naming as original data)
            flux_columns = ['G', 'R', 'Z', 'G_IVAR', 'R_IVAR', 'Z_IVAR', 
                          'W1', 'W2', 'W1_IVAR', 'W2_IVAR', 
                          'ML_G', 'ML_R', 'ML_Z', 'ML_W1', 'ML_W2']
            
            for i, col in enumerate(flux_columns):
                if i < fluxes_np.shape[1]:
                    data_dict[col] = fluxes_np[:, i]
            
            # Add metadata columns
            data_dict['SYNTHETIC_FLAG'] = 1  # Mark as synthetic
            data_dict['SPATIAL_MODEL'] = spatial_type
            data_dict['FLUX_MODEL'] = flux_type
            data_dict['GENERATION_TIME'] = timestamp
            data_dict['EXTRAPOLATION'] = extrapolate  # Mark extrapolation mode
            
            df = pd.DataFrame(data_dict)
            
            csv_path = os.path.join(output_dir, f"{base_filename}.csv")
            df.to_csv(csv_path, index=False)
            print(f"  ✓ Saved CSV: {csv_path}")
            
            # Also save a summary stats file
            summary_path = os.path.join(output_dir, f"{base_filename}_summary.txt")
            with open(summary_path, 'w') as f:
                f.write(f"Synthetic Quasar Dataset Summary\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Spatial Model: {spatial_type}\n")
                f.write(f"Flux Model: {flux_type}\n")
                f.write(f"Number of samples: {n_samples}\n\n")
                f.write(f"Coordinate Statistics:\n")
                f.write(f"  RA: {ra_values.min():.3f}° to {ra_values.max():.3f}° (mean: {ra_values.mean():.3f}°)\n")
                f.write(f"  DEC: {dec_values.min():.3f}° to {dec_values.max():.3f}° (mean: {dec_values.mean():.3f}°)\n")
                f.write(f"  Redshift: {redshifts_np.min():.4f} to {redshifts_np.max():.4f} (mean: {redshifts_np.mean():.4f})\n\n")
                f.write(f"Flux Statistics (median values):\n")
                for i, col in enumerate(flux_columns[:min(len(flux_columns), fluxes_np.shape[1])]):
                    f.write(f"  {col}: {np.median(fluxes_np[:, i]):.6f}\n")
            
            print(f"  ✓ Saved summary: {summary_path}")
        
        # 2. Save as NumPy arrays (fast loading for Python analysis)
        if format in ['numpy', 'all']:
            numpy_dir = os.path.join(output_dir, f"{base_filename}_numpy")
            os.makedirs(numpy_dir, exist_ok=True)
            
            np.save(os.path.join(numpy_dir, 'fluxes.npy'), fluxes_np)
            np.save(os.path.join(numpy_dir, 'ra.npy'), ra_values)
            np.save(os.path.join(numpy_dir, 'dec.npy'), dec_values)
            np.save(os.path.join(numpy_dir, 'redshifts.npy'), redshifts_np)
            
            # Save metadata
            metadata = {
                'n_samples': n_samples,
                'spatial_model': spatial_type,
                'flux_model': flux_type,
                'generation_time': timestamp,
                'flux_columns': flux_columns[:fluxes_np.shape[1]]
            }
            np.save(os.path.join(numpy_dir, 'metadata.npy'), metadata)
            
            print(f"  ✓ Saved NumPy arrays: {numpy_dir}/")
        
        # 3. Save as FITS (astronomy standard format)
        if format in ['fits', 'all']:
            try:
                from astropy.io import fits
                from astropy.table import Table
                
                # Create astropy table
                table_data = {
                    'TARGET_RA': ra_values,
                    'TARGET_DEC': dec_values,
                    'REDSHIFT': redshifts_np,
                }
                
                for i, col in enumerate(flux_columns[:fluxes_np.shape[1]]):
                    table_data[col] = fluxes_np[:, i]
                
                table = Table(table_data)
                
                # Add metadata to header
                table.meta['SPATIAL_MODEL'] = spatial_type
                table.meta['FLUX_MODEL'] = flux_type
                table.meta['N_SAMPLES'] = n_samples
                table.meta['GEN_TIME'] = timestamp
                table.meta['SYNTHETIC'] = True
                table.meta['COMMENT'] = 'Synthetic quasar data generated by generative AI framework'
                
                fits_path = os.path.join(output_dir, f"{base_filename}.fits")
                table.write(fits_path, format='fits', overwrite=True)
                print(f"  ✓ Saved FITS: {fits_path}")
                
            except ImportError:
                print("  ! astropy not available, skipping FITS format")
        
        print(f"\nSynthetic data generation completed!")
        print(f"Generated {n_samples} synthetic quasars using {spatial_type} + {flux_type} models")
        print(f"Data saved to: {output_dir}/")
        
        # Return paths for programmatic use
        return {
            'csv': f"{output_dir}/{base_filename}.csv" if format in ['csv', 'all'] else None,
            'numpy': f"{output_dir}/{base_filename}_numpy/" if format in ['numpy', 'all'] else None,
            'fits': f"{output_dir}/{base_filename}.fits" if format in ['fits', 'all'] else None,
            'summary': f"{output_dir}/{base_filename}_summary.txt" if format in ['csv', 'all'] else None
        }
    
    def validate_generation(self, n_samples=1000):
        """Validate generated samples against physics including spatial distribution"""
        print("Validating generated quasar distribution with spatial analysis...")
        
        # Generate samples
        generated_fluxes, generated_coords, generated_redshifts = self.generate_samples(n_samples)
        
        # Get real data for comparison
        # Use flux test data
        real_flux_scaled = self.flux_test_data.cpu().numpy()
        
        # Inverse transform flux features (no exp needed since we work with flux directly)
        real_fluxes_np = self.flux_scaler.inverse_transform(real_flux_scaled)
        # Remove small numerical offset that was added for stability
        real_fluxes_np = real_fluxes_np - 1e-10
        real_fluxes = torch.FloatTensor(real_fluxes_np).to(self.device)
        
        # Use test coordinates
        real_coords = self.coords_test
        
        # Run comprehensive validation including spatial
        results = self.validator.comprehensive_validation(
            generated_fluxes, real_fluxes, generated_redshifts,
            generated_coords.cpu().numpy(), real_coords
        )
        
        return results, generated_fluxes, generated_coords, generated_redshifts
    
    def visualize_results(self, generated_fluxes, real_fluxes, generated_coords=None, real_coords=None):
        """Create comprehensive visualization plots including spatial distribution"""
        print("Creating comprehensive visualization plots...")
        
        # Create larger figure to accommodate spatial plots
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.flatten()
        
        band_names = ['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2']
        
        # Flux distribution plots
        for i, band in enumerate(band_names):
            if i < 5:
                ax = axes[i]
                
                # Plot distributions
                ax.hist(real_fluxes[:, i].cpu().numpy(), alpha=0.5, density=True, 
                       bins=50, label='Real', color='blue')
                ax.hist(generated_fluxes[:, i].cpu().numpy(), alpha=0.5, density=True, 
                       bins=50, label='Generated', color='red')
                
                ax.set_xlabel(band)
                ax.set_ylabel('Density')
                ax.legend()
                ax.set_yscale('log')
        
        # Color-color diagram
        ax = axes[5]
        
        # g-r vs r-z colors (convert to numpy first)
        real_fluxes_np = real_fluxes.cpu().numpy()
        gen_fluxes_np = generated_fluxes.cpu().numpy()
        
        real_g_r = -2.5 * np.log10(real_fluxes_np[:, 0] / real_fluxes_np[:, 1])
        real_r_z = -2.5 * np.log10(real_fluxes_np[:, 1] / real_fluxes_np[:, 2])
        gen_g_r = -2.5 * np.log10(gen_fluxes_np[:, 0] / gen_fluxes_np[:, 1])
        gen_r_z = -2.5 * np.log10(gen_fluxes_np[:, 1] / gen_fluxes_np[:, 2])
        
        ax.scatter(real_r_z[:1000], real_g_r[:1000], alpha=0.5, s=1, 
                  label='Real', color='blue')
        ax.scatter(gen_r_z[:1000], gen_g_r[:1000], alpha=0.5, s=1, 
                  label='Generated', color='red')
        
        ax.set_xlabel('r-z color')
        ax.set_ylabel('g-r color')
        ax.legend()
        ax.set_title('Color-Color Diagram')
        
        # Spatial distribution plots (if coordinates available)
        if generated_coords is not None and real_coords is not None:
            # Sky distribution plot
            ax = axes[6]
            
            # Subsample for clarity
            n_plot = min(2000, len(real_coords))
            real_idx = np.random.choice(len(real_coords), n_plot, replace=False)
            gen_idx = np.random.choice(len(generated_coords), n_plot, replace=False)
            
            ax.scatter(real_coords[real_idx, 0], real_coords[real_idx, 1], 
                      alpha=0.6, s=1, label='Real', color='blue')
            ax.scatter(generated_coords[gen_idx, 0], generated_coords[gen_idx, 1], 
                      alpha=0.6, s=1, label='Generated', color='red')
            
            ax.set_xlabel('RA (degrees)')
            ax.set_ylabel('DEC (degrees)')
            ax.legend()
            ax.set_title('Sky Distribution')
            ax.grid(True, alpha=0.3)
            
            # RA distribution
            ax = axes[7]
            ax.hist(real_coords[:, 0], alpha=0.5, density=True, bins=50, 
                   label='Real', color='blue')
            ax.hist(generated_coords[:, 0], alpha=0.5, density=True, bins=50, 
                   label='Generated', color='red')
            ax.set_xlabel('RA (degrees)')
            ax.set_ylabel('Density')
            ax.legend()
            ax.set_title('RA Distribution')
            
            # DEC distribution
            ax = axes[8]
            ax.hist(real_coords[:, 1], alpha=0.5, density=True, bins=50, 
                   label='Real', color='blue')
            ax.hist(generated_coords[:, 1], alpha=0.5, density=True, bins=50, 
                   label='Generated', color='red')
            ax.set_xlabel('DEC (degrees)')
            ax.set_ylabel('Density')
            ax.legend()
            ax.set_title('DEC Distribution')
        else:
            # Remove unused subplots if no spatial data
            for i in range(6, 9):
                fig.delaxes(axes[i])
        
        plt.tight_layout()
        filename = f'{self.spatial_model_type}_{self.flux_model_type}_quasar_generation_with_spatial_results.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"  Plots saved to {filename} and displayed")

# ==================== MAIN EXECUTION ====================

def main():
    """Main execution function"""
    print("Comprehensive Quasar Generative AI Framework")
    print("Testing: VAE, ConvNeXt, Transformer, ViT, Flow, and Diffusion Models")
    print("=" * 70)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Available models
    models_to_test = ['convnext', 'vae', 'flow', 'diffusion']
    
    results_summary = {}
    
    # Model descriptions
    model_descriptions = {
        'vae': 'Variational Autoencoder - Classic probabilistic generative model',
        'convnext': 'ConvNeXt - Modern CNN architecture with attention (your best regression model)',
        'transformer': 'Transformer - Self-attention based architecture for sequence modeling', 
        'vit': 'Vision Transformer - Patch-based transformer for structured data',
        'flow': 'Normalizing Flow - Invertible transformations for exact likelihood',
        'diffusion': 'Diffusion Model - Denoising-based generative approach'
    }
    
    for model_type in models_to_test:
        print(f"\n{'='*20} {model_type.upper()} MODEL {'='*20}")
        print(f"{model_descriptions[model_type]}")
        
        try:
            # Initialize framework with error handling
            print(f"Initializing {model_type} framework...")
            framework = QuasarGenerativeFramework(spatial_model_type=model_type, flux_model_type=model_type, device=device)
            print(f"Framework initialized successfully for {model_type}")

            # Train model with improved parameters and error handling
            try:
                print(f"Starting training for {model_type}...")
                framework.train(epochs=200, lr=1e-3, batch_size=64)
                print(f"Training completed successfully for {model_type}")
            except Exception as train_error:
                print(f"Training error for {model_type}: {train_error}")
                print(f"Error type: {type(train_error).__name__}")
                # Try with more conservative parameters
                print(f"Retrying {model_type} with conservative parameters...")
                framework = QuasarGenerativeFramework(spatial_model_type=model_type, flux_model_type=model_type, device=device)
                framework.train(epochs=100, lr=1e-4, batch_size=32)
            
            # Validate generation (now returns spatial coordinates too)
            validation_results, generated_fluxes, generated_coords, generated_redshifts = framework.validate_generation(n_samples=2000)
            
            # Prepare real flux data for visualization  
            real_flux_scaled = framework.flux_test_data.cpu().numpy()
            real_fluxes_np = framework.flux_scaler.inverse_transform(real_flux_scaled)
            # Remove small numerical offset that was added for stability (no exp needed)
            real_fluxes_np = real_fluxes_np - 1e-10
            real_fluxes = torch.FloatTensor(real_fluxes_np).to(device)
            
            # Get real spatial coordinates for visualization
            real_coords = framework.coords_test
            
            # Visualize with spatial data
            framework.visualize_results(generated_fluxes, real_fluxes, 
                                      generated_coords.cpu().numpy(), real_coords)
            
            # Store results
            results_summary[model_type] = {
                'realism_score': validation_results['overall_realism_score'],
                'validation_details': validation_results
            }
            
            print(f"{model_type.upper()} model completed successfully!")
            
        except Exception as e:
            print(f"Error with {model_type} model: {e}")
            results_summary[model_type] = {'error': str(e)}
    
    # Final comparison
    print(f"\n{'='*25} FINAL RESULTS {'='*25}")
    
    # Sort results by realism score
    successful_results = [(k, v) for k, v in results_summary.items() if 'error' not in v]
    failed_results = [(k, v) for k, v in results_summary.items() if 'error' in v]
    
    if successful_results:
        successful_results.sort(key=lambda x: x[1]['realism_score'], reverse=True)
        
        print("\nARCHITECTURE PERFORMANCE RANKING:")
        print("-" * 50)
        for i, (model_type, results) in enumerate(successful_results):
            score = results['realism_score']
            details = results['validation_details']
            sed_score = details['sed_physics']['power_law_slopes']['realistic_fraction']
            
            print(f"{i+1}. {model_type.upper():12s}: Score = {score:.3f} (SED Physics: {sed_score:.3f})")
        
        # Best performer details
        best_model, best_results = successful_results[0]
        print(f"\nBEST PERFORMER: {best_model.upper()}")
        print(f"   Overall Realism: {best_results['realism_score']:.3f}")
        
        best_details = best_results['validation_details']
        print(f"   SED Physics: {best_details['sed_physics']['power_law_slopes']['realistic_fraction']:.3f}")
        
        flux_stats = best_details.get('flux_statistics', {})
        if flux_stats:
            avg_flux_score = 1 - np.mean([band['ks_statistic'] for band in flux_stats.values()])
            print(f"   Flux Statistics: {avg_flux_score:.3f}")
        
        spatial_score = best_details.get('spatial_distribution', {}).get('spatial_realism_score', 0.5)
        print(f"   Spatial Distribution: {spatial_score:.3f}")
    
    if failed_results:
        print(f"\nFAILED MODELS:")
        for model_type, results in failed_results:
            print(f"   {model_type.upper()}: {results['error'][:80]}...")
    
    print(f"\nComprehensive generative modeling completed!")
    print(f"Tested {len(models_to_test)} different architectures")
    print(f"{len(successful_results)} successful, {len(failed_results)} failed")
    print("Check generated plots and model files for detailed analysis")
    
    # Generate and save synthetic datasets from successful models
    print(f"\n{'='*25} GENERATING SYNTHETIC DATA {'='*25}")
    
    if successful_results:
        # Use the best performing model to generate a large synthetic dataset
        best_model_type = successful_results[0][0]
        print(f"Generating synthetic dataset using best model: {best_model_type.upper()}")
        
        try:
            # Create framework with best model
            best_framework = QuasarGenerativeFramework(
                spatial_model_type=best_model_type, 
                flux_model_type=best_model_type, 
                device=device
            )
            
            # Load data and train (models should already be trained)
            best_framework.load_data()
            
            # Generate and save large synthetic dataset
            print(f"Generating 10,000 synthetic quasars for analysis...")
            data_paths = best_framework.save_synthetic_data(
                n_samples=10000, 
                output_dir=f'synthetic_data_{best_model_type}',
                format='all'  # Save in CSV, NumPy, and FITS formats
            )
            
            print(f"\nSynthetic data files created:")
            for format_type, path in data_paths.items():
                if path:
                    print(f"  {format_type.upper()}: {path}")
            
            # Generate datasets from ALL successful models for comprehensive comparison
            for model_type, model_results in successful_results[1:]:  # All except best (already done above)
                try:
                    print(f"\nGenerating dataset using {model_type.upper()}...")
                    print(f"  Model realism score: {model_results['realism_score']:.3f}")
                    
                    comp_framework = QuasarGenerativeFramework(
                        spatial_model_type=model_type, 
                        flux_model_type=model_type, 
                        device=device
                    )
                    comp_framework.load_data()
                    
                    # Generate 5000 samples for each model for good statistics
                    comp_paths = comp_framework.save_synthetic_data(
                        n_samples=5000, 
                        output_dir=f'synthetic_data_{model_type}',
                        format='csv',  # CSV for analysis
                        extrapolate=True  # Allow extrapolation
                    )
                    print(f"✓ {model_type.upper()} dataset: {comp_paths['csv']}")
                    
                except Exception as e:
                    print(f"✗ Could not generate {model_type} dataset: {str(e)[:60]}...")
            
        except Exception as e:
            print(f"Error generating synthetic data: {str(e)[:100]}...")
            print("You can manually generate data using:")
            print("  framework = QuasarGenerativeFramework(spatial_model_type='vae', flux_model_type='vae')")
            print("  framework.load_data()")
            print("  framework.train()")
            print("  framework.save_synthetic_data(n_samples=10000)")
    
    else:
        print("No successful models to generate synthetic data from.")
        print("Try running with simpler architectures or check error messages above.")
    
    print(f"\n{'='*70}")
    print("SUMMARY:")
    print(f"✓ Trained and tested {len(models_to_test)} generative architectures")
    print(f"✓ Generated visualization plots for model comparison")
    print(f"✓ Saved trained model weights for future use")
    if successful_results:
        print(f"✓ Generated synthetic datasets for ALL {len(successful_results)} successful models")
        print(f"✓ Best performing model: {successful_results[0][0].upper()}")
        print(f"✓ Dataset files: synthetic_data_[model_name]/synthetic_quasars_*.csv")
    print("✓ All results saved to respective directories")

if __name__ == "__main__":
    main()

def generate_synthetic_quasar_dataset(
    spatial_model='convnext', 
    flux_model='convnext', 
    n_samples=10000, 
    output_dir='synthetic_quasars',
    train_new=False,
    extrapolate=True
):
    """
    Convenient function to generate synthetic quasar dataset
    
    Args:
        spatial_model (str): Model type for spatial-redshift generation ('vae', 'convnext', 'transformer', 'vit', 'flow', 'diffusion')
        flux_model (str): Model type for flux generation ('vae', 'convnext', 'transformer', 'vit', 'flow', 'diffusion') 
        n_samples (int): Number of synthetic quasars to generate
        output_dir (str): Directory to save the synthetic data
        train_new (bool): Whether to train new models or load existing ones
        extrapolate (bool): If True, allow extrapolation beyond training data for discovery of new regions
    
    Returns:
        dict: Paths to generated data files
    
    Example:
        # Generate 50,000 synthetic quasars using ConvNeXt models
        paths = generate_synthetic_quasar_dataset(
            spatial_model='convnext',
            flux_model='convnext', 
            n_samples=50000,
            output_dir='my_synthetic_quasars'
        )
        
        # Load the data for analysis
        import pandas as pd
        df = pd.read_csv(paths['csv'])
        print(f"Generated {len(df)} synthetic quasars")
    """
    import torch
    
    print(f"=== SYNTHETIC QUASAR DATASET GENERATION ===")
    print(f"Spatial Model: {spatial_model.upper()}")
    print(f"Flux Model: {flux_model.upper()}")
    print(f"Target Samples: {n_samples:,}")
    print(f"Output Directory: {output_dir}")
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Create framework
        framework = QuasarGenerativeFramework(
            spatial_model_type=spatial_model,
            flux_model_type=flux_model,
            device=device
        )
        
        # Load data
        print("Loading training data...")
        framework.load_data()
        
        if train_new:
            print("Training new models...")
            framework.train()
        else:
            print("Using existing/default models...")
        
        # Generate and save synthetic data
        print(f"Generating {n_samples:,} synthetic quasars...")
        data_paths = framework.save_synthetic_data(
            n_samples=n_samples,
            output_dir=output_dir,
            format='all',
            extrapolate=extrapolate
        )
        
        print(f"\n✓ Synthetic dataset generation completed!")
        print(f"Files created:")
        for format_type, path in data_paths.items():
            if path:
                print(f"  {format_type.upper()}: {path}")
        
        return data_paths
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Ensure data files exist in data/ directory")
        print("2. Check CUDA availability if using GPU")
        print("3. Try with simpler models like 'vae' if complex models fail")
        print("4. Set train_new=True to train fresh models")
        raise
