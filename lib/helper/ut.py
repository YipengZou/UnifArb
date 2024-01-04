import pandas as pd
from typing import Union
import numpy as np

def to_local_datetime(dt: Union[pd.Timestamp, str]) -> pd.Timestamp:
    if isinstance(dt, str):
        dt = pd.Timestamp(dt)
    
    return dt.tz_localize("Asia/Hong_Kong")


def calc_distance_weight(
        arr: Union[pd.Series, np.ndarray, list]
)-> np.ndarray:
    """Calculate non-linear weights using exponential function and distance to the last number"""
    last_index = len(arr) - 1
    distances = np.abs(np.arange(len(arr)) - last_index)  # Calculate the distance of each number to the last number
    normalized_distances = distances / np.max(distances)  # Normalize the distances
    weights = np.exp(-normalized_distances)  # Calculate non-linear weights using exponential function
    normalized_weights = weights / np.sum(weights)  # Normalize weights to sum up to 1
    return normalized_weights

def weighted_avg_and_std(values: pd.Series, weights: pd.Series):
    """
    Return the weighted average and standard deviation.

    They weights are in effect first normalized so that they 
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values)
    # Fast and numerically precise:
    variance = np.average((values - average)**2, weights = weights)
    return (average, np.sqrt(variance))