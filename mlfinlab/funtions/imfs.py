import emd
import pandas as pd


def decompose_imfs(z: pd.Series, column: str = "Close", max_imf=-1):
    """
    Decomposes a time series into its intrinsic mode functions (IMFs) using the Empirical Mode Decomposition (EMD) method.

    Args:
        z (pd.Series): The time series to be decomposed.
        column (str, optional): The column name of the time series. Defaults to "Close".
        max_imf (int, optional): The maximum number of IMFs to be computed. Defaults to -1, which means all IMFs will be computed.

    Returns:
        pd.DataFrame: A DataFrame containing the IMFs as columns, with the time series index as the row index.
    """
    imfs = emd.sift.sift(z.values, max_imfs=max_imf)
    imfnum = imfs.shape[1]
    columns = [f"{column}_{i}" for i in range(imfnum)]
    return pd.DataFrame(imfs, columns=columns, index=z.index)
