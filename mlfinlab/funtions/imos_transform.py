import pandas as pd
import numpy as np


def imos_transform(s: pd.Series, H: int = 256):
    """
    Applies the IMOS (Inverse Moving Sum) transform to a given pandas Series.

    Parameters:
    s (pd.Series): The input time series data.
    H (int): The window size for calculating the rolling sum. Default is 256.

    Returns:
    pd.Series: The transformed time series data.
    pd.Series: The rolling norm values used for normalization.
    """
    norm = s.rolling(H).apply(lambda x: np.linalg.norm(x, ord=2)).dropna()
    normalized = s / norm
    return normalized.rolling(H).sum(), norm
