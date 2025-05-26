import pandas as pd
import numpy as np
from pymsis import msis as pymsis
import random
import torch
import json
import math
import os
import joblib
from copy import deepcopy
import torch.optim as optim
from pathlib import Path
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from tqdm.notebook import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from feat_eng import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")
seed_everything()



def orbit_mean(rho_inst, a_km, step_sec=600):
    """rho_inst : (432,) instantaneous density
       a_km      : semi‑major axis from initial_states (km)"""
    mu = 3.986004418e5      # km³ s⁻²
    T  = 2*np.pi*np.sqrt((a_km**3)/mu)      # seconds
    
    w = int(round(T/step_sec))             # samples per orbit  (≈9‑10)
    # 1‑D convolution with a uniform kernel
    kernel = np.ones(w)/w
    # 'valid' keeps length = 432‑w+1, so pad at the front to match 432
    mm = np.convolve(rho_inst, kernel, mode='valid')
    pad = np.full(w-1, mm[0])               # repeat first mean
    return np.concatenate([pad, mm])


def metric(predictions, targets, bad_flags, epsilon=1e-5):
    """
    Calculate weighted RMSE metric properly handling bad flags.
    
    Parameters:
    -----------
    predictions : numpy array of shape (432,)
        Model predictions
    targets : numpy array of shape (432,)
        Ground truth values
    bad_flags : numpy array of shape (432,)
        Array indicating bad values (1 = bad/invalid, 0 = good)
    epsilon : float
        Small value for weight calculation
    """
    # Ensure inputs are numpy arrays
    pred_values = np.array(predictions)
    true_values = np.array(targets)
    bad_flags = np.array(bad_flags)
    
    # Calculate timestep weights
    sequence_length = len(true_values)
    T = sequence_length * 10 * 60  # Total duration in seconds (10-min intervals)
    gamma = -np.log(epsilon) / T
    timesteps = np.arange(sequence_length) * 10 * 60
    weights = np.exp(-gamma * timesteps)
    
    # Zero out weights for bad points (create mask where 0 = valid)
    valid_mask = (bad_flags == 0).astype(float)
    masked_weights = weights * valid_mask
    
    # Normalize weights to sum to 1 after masking
    if np.sum(masked_weights) > 0:
        masked_weights = masked_weights / np.sum(masked_weights)
    
    # Calculate squared errors
    squared_errors = (pred_values - true_values) ** 2
    
    # Calculate weighted RMSE
    weighted_sum_squared_errors = np.sum(masked_weights * np.nan_to_num(squared_errors))
    weighted_rmse = np.sqrt(weighted_sum_squared_errors)
    
    return weighted_rmse


class DirectWeightedRMSELoss(nn.Module):
    def __init__(self, sequence_length=FORECAST_SIZE, epsilon=1e-5):
        """
        Parameters:
          sequence_length: Number of timesteps in each sample.
          epsilon: Small value for numerical stability in the weight computation.
        """
        super(DirectWeightedRMSELoss, self).__init__()
        T = sequence_length * 10 * 60
        gamma = -torch.log(torch.tensor(epsilon)) / T
        timesteps = torch.arange(sequence_length, dtype=torch.float32) * 10 * 60
        weights = torch.exp(-gamma * timesteps)
        weights = weights / weights.sum()  # initially normalized over all timesteps
        self.register_buffer('weights_buffer', weights)

    def forward(self, predictions, targets, bad_flag):
        """
        predictions: Tensor of shape (batch_size, sequence_length)
        targets: Tensor of the same shape as predictions.
        bad_flag: Tensor of the same shape, with 1 indicating an interpolated (or otherwise
                  less-trustworthy) value, and 0 indicating a valid value.
        """
        # nans to zero
        targets = torch.where(torch.isnan(targets), torch.zeros_like(targets), targets)
        
        squared_errors = (predictions - targets) ** 2
        # Expand weights to the batch dimension:
        weights = self.weights_buffer.unsqueeze(0).expand_as(squared_errors)
        # Zero out the weights corresponding to bad data:
        weights = weights * (bad_flag == 0).float()
        
        weighted_errors = squared_errors * weights
        weighted_sum = weighted_errors.sum(dim=1)
        loss = torch.sqrt(weighted_sum).mean()
        return loss

class WeightedLogCoshRatioLoss(nn.Module):
    """
    Time–weighted, robust loss for log-ratio targets.
    * predictions, targets : log-ratio tensors [B, T]
    * bad_flag             : 1 = unreliable sample (weight → 0)
    """
    def __init__(self, sequence_length=432, epsilon=1e-5):
        super().__init__()
        # --- same exponential-time weighting you already use -----------------
        T  = sequence_length * 10 * 60          # horizon length in seconds
        γ  = -torch.log(torch.tensor(epsilon)) / T
        t  = torch.arange(sequence_length, dtype=torch.float32) * 10 * 60
        w  = torch.exp(-γ * t)
        w  = w / w.sum()                        # normalise
        self.register_buffer("w", w)            # shape [T]
        # --------------------------------------------------------------------

    def forward(self, pred, target, bad_flag):
        # nan-safe targets
        target = torch.where(torch.isnan(target), torch.zeros_like(target), target)

        diff    = pred - target                 # [B, T]
        # log-cosh:  log( (e^x + e^{-x}) / 2 )
        loss_t  = torch.log(torch.cosh(diff))   # robust per-time loss
        w       = self.w.unsqueeze(0).expand_as(loss_t)
        w       = w * (bad_flag == 0).float()   # zero-out unreliable points
        # batch-wise weighted RMSE analogue in log space
        loss_b  = (w * loss_t).sum(dim=1).sqrt()
        return loss_b.mean()

class SatelliteDataset(Dataset):
    def __init__(self, omni_df, initial_states_df, target_df, physics, 
                 omni_features, state_features, physics_features, 
                 target_col_name="interpolated", 
                 bad_flag_col_name="bad_flag"):
        self.satellite_ids = sorted(initial_states_df["File ID"].unique())
        self.all_targets = {}
        self.all_physical_densities = {}
        self.all_bad_flags = {}
        self.all_features = {}
        for sat_id in tqdm(self.satellite_ids):
            self.all_features[sat_id] = [
                omni_df[omni_df["file_id"]==sat_id][omni_features].values.astype(np.float32), 
                initial_states_df[initial_states_df["File ID"]==sat_id][state_features].values.astype(np.float32)
            ]
            self.all_targets[sat_id] = target_df[target_df["file_id"]==sat_id][target_col_name].values.astype(np.float32)
            self.all_bad_flags[sat_id] = target_df[target_df["file_id"]==sat_id][bad_flag_col_name].values.astype(np.float32)
            self.all_physical_densities[sat_id] = physics[physics["file_id"]==sat_id][physics_features].values.astype(np.float32)
    
    def __len__(self):
        return len(self.satellite_ids)
    
    def __getitem__(self, idx):
        sat_id = self.satellite_ids[idx]
        omni_features, state_features = self.all_features[sat_id]
        return (omni_features, 
                state_features, 
                self.all_targets[sat_id], 
                self.all_physical_densities[sat_id], 
                self.all_bad_flags[sat_id], 
                sat_id)


class SatelliteDatasetEval(Dataset):
    def __init__(self, omni_df, initial_states_df, physics, 
                 omni_features, state_features, physics_features):
        self.satellite_ids = sorted(initial_states_df["File ID"].unique())
        self.all_physical_densities = {}
        self.all_features = {}
        self.TARGET_LEN = OMNI_SEQUENCE_SIZE
        for sat_id in tqdm(self.satellite_ids):
            omni_arr = omni_df[omni_df["file_id"]==sat_id][omni_features].values.astype(np.float32)
            n_missing = self.TARGET_LEN - omni_arr.shape[0]
            if n_missing > 0: # pad (zeros)
                print("edge case")
                pad = np.repeat(omni_arr[-1:], n_missing, axis=0)
                omni_arr = np.vstack((omni_arr, pad))
            elif n_missing < 0:                                 # truncate
                print("edger case")
                omni_arr = omni_arr[-self.TARGET_LEN:]
            
            self.all_features[sat_id] = [
                omni_arr, 
                initial_states_df[initial_states_df["File ID"]==sat_id][state_features].values.astype(np.float32)
            ]
            self.all_physical_densities[sat_id] = physics[physics["file_id"]==sat_id][physics_features].values.astype(np.float32)
    
    def __len__(self):
        return len(self.satellite_ids)
    
    def __getitem__(self, idx):
        sat_id = self.satellite_ids[idx]
        omni_features, state_features = self.all_features[sat_id]
        return (omni_features, 
                state_features, 
                self.all_physical_densities[sat_id], 
                sat_id)

class SatelliteDensityModel(nn.Module):
    """
    Model specifically designed for the provided tensor shapes:
    - omni: [32, OMNI_SEQUENCE_SIZE, 104]
    - static: [32, 1, 222]
    - densities: list of 4 tensors, each [32, 432]
    """
    def __init__(self,
                 omni_dim=104,
                 static_dim=222,
                 hidden_dim=256,
                 phys_len=432,
                 n_phys=4,  # Based on the list length of 4
                 dropout=0.2):
        super().__init__()
        
        # ===== OMNI Sequence Processing =====
        self.omni_gru = nn.GRU(
            input_size=omni_dim,
            hidden_size=hidden_dim//2,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # ===== Static Feature Processing =====
        # First squeeze out the middle dimension
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.GELU()
        )
        
        # ===== Physics Model Processing =====
        # Create a gating mechanism for physics models
        self.physics_gate = nn.Sequential(
            nn.Linear(hidden_dim//2, n_phys),
            nn.Sigmoid()
        )
        
        # Multi-scale CNN processing for physics data
        # Critical: Input shape will be [B, C, T] where C=n_phys and T=phys_len
        self.physics_convs = nn.ModuleList()
        dilations = [1, 2, 4]
        kernel_sizes = [3, 5, 7]
        
        for k_size, dilation in zip(kernel_sizes, dilations):
            padding = ((k_size - 1) * dilation) // 2
            self.physics_convs.append(nn.Sequential(
                nn.Conv1d(n_phys, hidden_dim//4, kernel_size=k_size, padding=padding, dilation=dilation),
                nn.BatchNorm1d(hidden_dim//4),
                nn.GELU()
            ))
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # ===== Feature Fusion =====
        cnn_features = (hidden_dim//4) * len(self.physics_convs)
        self.fused_dim = hidden_dim + hidden_dim//2 + cnn_features
        
        self.fusion_norm = nn.LayerNorm(self.fused_dim)
        
        # ===== Output Networks =====
        self.regular_head = nn.Sequential(
            nn.Linear(self.fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, phys_len)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, omni_seq, static_x, phys_list):
        # ===== Process OMNI sequence =====
        # Input shape: [B, 1440, 104]
        _, h_omni = self.omni_gru(omni_seq)
        # Extract final hidden states (bidirectional) - shape: [B, hidden_dim]
        omni_embed = torch.cat([h_omni[-2], h_omni[-1]], dim=-1)
        
        # ===== Process static features =====
        # Input shape: [B, 1, 222]
        # First, squeeze out the middle dimension
        static_x = static_x.squeeze(1)  # Now shape: [B, 222]
        static_embed = self.static_mlp(static_x)
        
        # ===== Process physics models =====
        # Input shape: list of 4 tensors, each [B, 432]
        # Stack along a new dimension to get [B, 4, 432]
        phys_stacked = torch.stack(phys_list, dim=1)
        
        # Apply gating from static features
        gates = self.physics_gate(static_embed).unsqueeze(-1)  # [B, 4, 1]
        phys_gated = phys_stacked * gates  # Shape: [B, 4, 432]
        
        # Multi-scale CNN processing
        cnn_outputs = []
        for conv in self.physics_convs:
            # Input shape to conv: [B, 4, 432]
            conv_out = conv(phys_gated)
            pooled = self.global_pool(conv_out).squeeze(-1)
            cnn_outputs.append(pooled)
        
        # Concatenate CNN outputs
        phys_embed = torch.cat(cnn_outputs, dim=1)
        
        # ===== Fuse all embeddings =====
        fused = torch.cat([omni_embed, static_embed, phys_embed], dim=1)
        fused_norm = self.fusion_norm(fused)
        
        # ===== Generate predictions =====
        return self.regular_head(fused_norm)


class SatelliteDensityModelV4(nn.Module):
    """
    Inputs:
      omni_seq   : [B, T1, D1] (OMNI sequence)
      static_x   : [B, S]       (static features)
      phys_list  : list of 5 tensors [B, 432] (logρ_MSIS, logρ_JB, logρ_DTM, diff_JB-MSIS, diff_DTM-MSIS)
    Output:
      Δρ_pred    : [B, 432]
    """
    def __init__(self,
                 omni_dim,
                 static_dim,
                 hidden=128,
                 phys_len=FORECAST_SIZE,
                 n_phys=5,
                 dropout=0.2):
        super().__init__()
        # OMNI GRU encoder (bidirectional)
        self.omni_gru = nn.GRU(omni_dim, hidden//2, num_layers=2,
                               dropout=dropout, batch_first=True, bidirectional=True)
        
        # Static MLP
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, hidden//2),
            nn.GELU(),
            #nn.Dropout(0.1),
            nn.LayerNorm(hidden//2)
        )
        # Physics + dynamic channels
        total_channels = n_phys
        # Self-attention over time on phys+dynamics
        self.phys_attn = nn.MultiheadAttention(embed_dim=n_phys, num_heads=n_phys, batch_first=True)
        # Multi-scale dilated CNN paths
        self.phys_convs = nn.ModuleList([
            nn.Conv1d(total_channels, hidden//4, kernel_size=3, padding=1, dilation=1),
            nn.Conv1d(total_channels, hidden//4, kernel_size=5, padding=2, dilation=2),
            nn.Conv1d(total_channels, hidden//4, kernel_size=9, padding=4, dilation=4)
        ])
        self.phys_pool = nn.AdaptiveAvgPool1d(1)
        # Gating for physics sources from static
        self.gate_layer = nn.Sequential(
            nn.Linear(hidden//2, total_channels),
            nn.Sigmoid()
        )
        # Final fusion head
        fused_dim = hidden + hidden//2 + 3*(hidden//4)
        
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, phys_len)
        )

    # ------------------------------------------------------------------

    def forward(self, omni_seq, static_x, phys_list):
        # OMNI encoding
        _, h_omni = self.omni_gru(omni_seq)
        omni_embed = torch.cat([h_omni[-2], h_omni[-1]], dim=-1)
        # Static embedding
        static_embed = self.static_mlp(static_x).squeeze(1)
        # Stack physics + dynamic channels -> [B, C, T]
        phys = torch.stack(phys_list, dim=1)
        # Compute gates from static -> [B, C, 1]
        gates = self.gate_layer(static_embed).unsqueeze(-1)
        phys = phys * gates
        # Self-attention over time: permute to [B, T, C]
        phys_t = phys.permute(0, 2, 1)
        phys_attn_out, _ = self.phys_attn(phys_t, phys_t, phys_t)
        phys = phys_attn_out.permute(0, 2, 1)  # back to [B, C, T]
        # Multi-scale CNN and pooling
        conv_outs = []
        for conv in self.phys_convs:
            x = conv(phys)
            x = F.gelu(x)
            conv_outs.append(self.phys_pool(x).squeeze(-1))
        phys_embed = torch.cat(conv_outs, dim=1)
        # Fuse all features
        fused = torch.cat([omni_embed, static_embed, phys_embed], dim=1)
        return self.head(fused)

