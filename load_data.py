import random
import pandas as pd
import numpy as np
from typing import Union, Callable


def load_data(
    data_paths: Union[list[str], str],
    voltages_start: int = 0,
    voltages_end: int = 13,
    output_values_start: int = 14,
    output_values_end: int = 26,
):
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
    data_paths = [data_paths] if isinstance(data_paths, str) else data_paths

    data = [pd.read_csv(path, header=None, chunksize=2000) for path in data_paths]
    all_chunks = []
    for ds in data:
        for data_chunk in ds:
            all_chunks.append(data_chunk)
    # Combine all data first
    full_data = pd.concat(all_chunks, ignore_index=True)
    # Apply numeric conversion and drop NaN rows from the COMPLETE dataset
    full_data_numeric = full_data.apply(pd.to_numeric, errors="coerce").dropna()

    # Now split into voltages and output_values - they'll have the same number of rows
    voltages = full_data_numeric.iloc[:, voltages_start : voltages_end + 1].to_numpy()
    output_values = full_data_numeric.iloc[
        :, output_values_start : output_values_end + 1
    ].to_numpy()

    return voltages, output_values


def trim_outliers(
    raw_voltages,
    raw_outputs,
    trim_threshold: Union[float, list[float]],
    trim_mode: str = "z-score",
):
    if trim_mode == "z-score":
        scores = np.abs(
            (raw_outputs - np.mean(raw_outputs, axis=0)) / np.std(raw_outputs, axis=0)
        )
    if trim_mode == "iqr":
        q1 = np.percentile(raw_outputs, 25, axis=0)
        q3 = np.percentile(raw_outputs, 75, axis=0)
        iqr = q3 - q1
        scores = np.abs(raw_outputs - q1) / iqr

    # Create mask for rows without outliers (no column has z-score > threshold)
    outlier_mask = np.any(scores > trim_threshold, axis=1)
    clean_mask = ~outlier_mask

    # Apply the mask to both arrays to keep them aligned
    voltages = raw_voltages[clean_mask]
    outputs = raw_outputs[clean_mask]
    outlier_voltages = raw_voltages[outlier_mask]
    outlier_outputs = raw_outputs[outlier_mask]

    return voltages, outputs, outlier_voltages, outlier_outputs


def uniform_metric_distribution(
    voltages: np.ndarray,
    outputs: np.ndarray,
    metric: Callable[[np.ndarray], np.ndarray],
    n_bins: int = 7,
    max_metric: float = 600.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Prune input voltages and outputs such that the resultant distribution of metrics,
    `metric(outputs)`, is uniform. The number of resultant data points is the product of
    `n_bins` and the length of the smallest bin.

    Args
    ----
    voltages: np.ndarray
        raw set of voltages used to generate `outputs`
    outputs: np.ndarray
        raw simulation output data; must have the same length as voltages
    metric: Callable[[np.ndarray], np.ndarray]
        function which takes `outputs` as its sole argument and returns a vector with
        scalar values for each output
    n_bins: int
        number of bins used to digitize metrics prior to uniform sampling
    max_metric: float
        maximum metric considered in digitization. The raw metric distribution is
        typically very positively skewed. You'll want to ignore large values or most of
        the data will go into the smallest bin.

    Returns
    -------
    uniform_voltages: np.ndarray
        random sampling of provided voltages such that the associated outputs produce a
        uniform set of metrics
    uniform_outputs: np.ndarray
        random sampling of provided outputs such that `metric(uniform_outputs)` is a
        uniform distribution
    """
    assert (
        outputs.shape[0] == voltages.shape[0]
    ), "voltages and outputs have different lengths"

    metrics = metric(outputs)
    bin_edges = np.linspace(metrics.min(), max_metric, n_bins)
    indices = np.digitize(metrics, bin_edges)

    voltages_list: list = voltages.tolist()
    outputs_list: list = outputs.tolist()
    uniform_voltages = []
    uniform_outputs = []
    bin_empty = False
    while not bin_empty:
        for i in range(1, n_bins + 1):
            matches = np.where(indices == i)[0]
            if matches.size == 0:
                bin_empty = True
            else:
                random_match = random.choice(matches)

                indices = np.delete(indices, random_match)
                uniform_voltages.append(voltages_list.pop(random_match))
                uniform_outputs.append(outputs_list.pop(random_match))

    return np.array(uniform_voltages), np.array(uniform_outputs)
