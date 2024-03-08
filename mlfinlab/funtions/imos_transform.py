import pandas as pd
import numpy as np


# NOTE: 最初の255のデータが欠落するので逆変換ができない
# from sklearn.base import TransformerMixin, BaseEstimator
# class ImosTransformer(TransformerMixin, BaseEstimator):
#     def __init__(self, *, H: int = 256):
#         self._H = H

#     def fit(self, X, y=None):
#         self._norm = (
#             pd.DataFrame(X)
#             .rolling(self._H)
#             .apply(lambda x: np.linalg.norm(x, ord=2))
#             .dropna()
#         )
#         return self

#     def transform(self, X):
#         return (X / self._norm).rolling(self._H).sum()

#     def inverse_transform(self, X):
#         return X * self._norm


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
