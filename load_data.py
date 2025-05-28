import pandas as pd
import numpy as np
from typing import Tuple, Optional, Callable,Union

def load_data(data_paths:list[str], voltages_start:int=None ,voltages_end:int=13, output_values_start: int=14, output_values_end: int=None, trim_mode: str='z-score' , trim_threshold: Union[float, list[float]]=None):
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



        # Create mask for rows without outliers (no column has z-score > threshold)
        outlier_mask = np.any(scores > trim_threshold, axis=1)
        clean_mask = ~outlier_mask

        # Apply the mask to both arrays to keep them aligned
        voltages= voltages[clean_mask]
        output_values= output_values[clean_mask]


    


    return voltages, output_values