from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import random
from model import *
from load_data import load_data
def run_inference(predictor: ElectronOpticsPredictor=None, batch_size: int = 32, use_training_ds: bool = False,shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:


    if predictor is None:
        raise ValueError("Predictor instance must be provided for inference.")

    
    dataset = predictor.train_ds if use_training_ds else predictor.val_ds
    ds_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    #Run inference on validation loader
    predictor.model.eval()  # Set model to evaluation mode
    all_predictions_norm = []
    all_values_norm = []

    with torch.no_grad():  # No gradients needed for inference
        for batch_voltages, batch_values in ds_loader:
            # Move data to device
            batch_voltages = batch_voltages.to(predictor.device)
            
            # Forward pass
            outputs = predictor.model(batch_voltages)
            
            # Store predictions and ground truth
            all_predictions_norm.append(outputs.cpu().numpy())
            all_values_norm.append(batch_values.numpy())

    # 4. Concatenate all batches
    all_predictions_norm = np.vstack(all_predictions_norm)
    all_values_norm = np.vstack(all_values_norm)
    all_predictions = predictor._denormalize_values(all_predictions_norm)
    all_true_values = predictor._denormalize_values(all_values_norm)

    return all_predictions, all_true_values


def plot_inference_comparison(predictor, n_samples: int=None, subplot_shape: Tuple[int, ...]=(3,5),figsize=(20, 10),**kwargs):

    """Simple wrapper for inference plotting."""
    all_predictions, all_true_values = run_inference(predictor,**kwargs)
    fig, ax = plt.subplots(*subplot_shape, figsize=figsize)
    ax = ax.flatten()
    n_samples=len(all_predictions) if n_samples is None else n_samples
    x_range = np.arange(n_samples)
    for v in range(all_predictions.shape[1]):  # auto-detect n_output_values
        ax[v].scatter(x_range, all_predictions[:n_samples, v], 
                     label='Predictions', color='red', s=100, marker='x')
        ax[v].scatter(x_range, all_true_values[:n_samples, v], 
                     label='True Values', color='black', s=100)
        ax[v].set_title(f'Output {v+1}')
        ax[v].legend()
    
    plt.tight_layout()
    return fig, ax, all_predictions, all_true_values





def trim_hist(
    file_list: List[str],
    output_values_start: int,
    output_values_end: int,
    trim_threshold: float = 0.1,
    figsize: Tuple[int, int] = (20, 15),
    subplot_shape: Tuple[int, int] = (5, 4),
    colors: Optional[Dict[str, str]] = None,
    hist_bins: int = 100,
    point_size: int = 100,
    alpha: float = 0.5,
    plot_output: bool = False,
    **kwargs: Any
) -> Tuple[Any, Any, Any, Any, plt.Figure, np.ndarray]:
    """
    Load data with and without cleaning, then plot comparison visualizations.
    
    Args:
        file_list: List of CSV files to load
        output_values_start: Start index for output values
        output_values_end: End index for output values
        trim_threshold: Threshold for data cleaning
        figsize: Figure size
        subplot_shape: Shape of subplot grid (rows, cols)
        colors: Color scheme dict with keys like 'clean', 'unclean', 'output'
        hist_bins: Number of histogram bins
        point_size: Size of scatter plot points
        alpha: Transparency for histograms
    
    Returns:
        voltages_unclean, output_values_unclean, voltages, output_values, fig, ax
    """
    # Default colors
    if colors is None:
        colors = {
    
            'unclean': 'black',
            'clean': 'blue', 
        }
    
    # Load data - unclean and clean versions
    voltages_unclean, output_values_unclean = load_data(
        file_list, 
        output_values_start=output_values_start, 
        output_values_end=output_values_end,**kwargs
    )
    
    voltages, output_values = load_data(
        file_list,
        output_values_start=output_values_start, 
        output_values_end=output_values_end,
        trim_threshold=trim_threshold,**kwargs
    )
    
    # Create subplots
    fig, ax = plt.subplots(*subplot_shape, figsize=figsize)
    ax = ax.flatten()

    
    # Plot voltage histograms (starting from subplot after outputs)
    n_voltage_channels = voltages.shape[1]
   
    
    for v in range(n_voltage_channels):
            # Make sure we don't exceed subplot count
            ax[v].hist(
                voltages_unclean[:, v], 
                bins=hist_bins, 
                color=colors['unclean'], 
                alpha=alpha, 
                label='Uncleaned Voltages'
            )
            ax[v].hist(
                voltages[:, v], 
                bins=hist_bins, 
                color=colors['clean'], 
                alpha=alpha, 
                label='Cleaned Voltages'
            )
            ax[v].set_title(f'Voltage Channel {v+1}')
            ax[v].legend()
    if plot_output:
        n_output_values = output_values.shape[1]
        for v in range(n_output_values):
                # Make sure we don't exceed subplot count
                ax[v+n_voltage_channels].hist(
                    output_values_unclean[:, v], 
                    bins=hist_bins, 
                    color=colors['unclean'], 
                    alpha=alpha, 
                    label='Uncleaned outputs'
                )
                ax[v+n_voltage_channels].hist(
                    output_values[:, v], 
                    bins=hist_bins, 
                    color=colors['clean'], 
                    alpha=alpha, 
                    label='Cleaned outputs'
                )
                ax[v+n_voltage_channels].set_title(f'Output {v+1}')
                ax[v+n_voltage_channels].legend()
    
    plt.tight_layout()
    
    # Calculate and print retention percentage
    retention_pct = (output_values.shape[0] / output_values_unclean.shape[0]) * 100
    print(f'{retention_pct:.1f}% of data retained after cleaning')
    
    return fig, ax, voltages_unclean, output_values_unclean, voltages, output_values






def trim_scatter(data_paths:list[str], voltages_start:int=None ,voltages_end:int=13, 
                output_values_start: int=14, output_values_end: int=None, 
                trim_mode: str='z-score' , trim_threshold: float=None, 
                figsize: Tuple[int, int] = (20, 15),
                subplot_shape: Tuple[int, int] = (5, 4),
                colors: Optional[Dict[str, str]] = None):
    """
    Load data from multiple CSV files, convert to numeric, and split into voltages and output values.

    Args:
        data_paths (list[str]): List of paths to CSV files.
        n_voltages (int): Number of voltage columns.
        output_values_start (int): Starting index for output values.
        output_values_end (int): Ending index for output values. If None, will use the last column.
        trim_mode (str): Mode for outlier trimming ('z-score' or 'iqr').
        trim_threshold (float): Threshold for outlier trimming.
    Returns:
        tuple: Tuple containing two numpy arrays: voltages and output values.
    """




    data=[pd.read_csv(path, header=None, chunksize=2000) for path in data_paths]
    all_chunks = []
    for ds in data:
        for data_chunk in ds:   
            all_chunks.append(data_chunk)
    # Combine all data first
    full_data = pd.concat(all_chunks, ignore_index=True)
    # Apply numeric conversion and drop NaN rows from the COMPLETE dataset
    full_data_numeric = full_data.apply(pd.to_numeric, errors='coerce').dropna()

    # Now split into voltages and output_values - they'll have the same number of rows
    voltages = full_data_numeric.iloc[:, voltages_start:voltages_end+1].to_numpy()
    output_values = full_data_numeric.iloc[:, output_values_start: output_values_end+1].to_numpy()
    
    if trim_threshold is not None:
        if trim_mode == 'z-score':
            scores = np.abs((output_values - np.mean(output_values, axis=0)) / np.std(output_values, axis=0))
        if trim_mode == 'iqr':
            q1 = np.percentile(output_values, 25, axis=0)
            q3 = np.percentile(output_values, 75, axis=0)
            iqr = q3 - q1
            scores = (np.abs(output_values - q1) / iqr)


        # trim_threshold= [trim_threshold] * output_values.shape[1] if isinstance(trim_threshold, (int, float)) else trim_threshold
        # Create mask for rows without outliers (no column has z-score > threshold)
        outlier_mask = np.any(scores > trim_threshold, axis=1)
        
        # Apply the mask to both arrays to keep them aligned
        voltages_masked= voltages.copy().astype(float)
        output_values_masked= output_values.copy().astype(float)
        voltages_masked[outlier_mask]= np.nan
        output_values_masked[outlier_mask]= np.nan

    fig,ax=plt.subplots(*subplot_shape, figsize=figsize)
    ax = ax.flatten()


    for v in range(output_values.shape[1]):  
        output_value= output_values[:,v]
        output_value=np.random.permutation(output_value)  # Randomize order for better visibility
        ax[v].scatter(np.arange(len(output_values)), output_value, label='full', color='k',s=100,marker='o')
    for v in range(output_values_masked.shape[1]):  
        output_value_masked= output_values_masked[:,v]
        output_value_masked=np.random.permutation(output_value_masked)  # Randomize order for better visibility
        ax[v].scatter(np.arange(len(output_values_masked)), output_value_masked, label='trimmed', color='b',s=10,marker='o')
        ax[v].set_title(f'Output {v+1}')
        ax[v].legend()
        # ax[v].axhline(np.mean(output_values[v], axis=0)+trim_threshold[v]*np.std(output_values, axis=0)[v], color='red', linestyle='--', linewidth=1)
        # ax[v].axhline(np.mean(output_values[v], axis=0)-trim_threshold[v]*np.std(output_values, axis=0)[v], color='red', linestyle='--', linewidth=1)
        
    
    plt.tight_layout()



    return fig, ax, voltages, output_values, voltages_masked, output_values_masked



def normalize_data(data: np.ndarray, scaler=None):
    """Normalize data using min-max scaling"""
    if scaler is None:
        scaler = {
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0)
        }
    
    # Avoid division by zero
    range_vals = scaler['max'] - scaler['min']
    range_vals[range_vals == 0] = 1
    
    normalized = (data - scaler['min']) / range_vals
    return normalized, scaler

def denormalize_data(normalized_data: np.ndarray, scaler: Dict[str, np.ndarray]):
    """Denormalize data using min-max scaling"""
    # Avoid division by zero
    range_vals = scaler['max'] - scaler['min']
    range_vals[range_vals == 0] = 1
    
    denormalized = normalized_data * range_vals + scaler['min']
    return denormalized