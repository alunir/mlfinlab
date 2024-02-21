import pandas as pd
import numpy as np


def pct(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the percentage change of OHLC prices and volume in a DataFrame.
    CAUTION. this function (pct_change indeed) is depending on the initial value.

    Args:
        df (pd.DataFrame): Input DataFrame containing OHLC prices and volume.

    Returns:
        pd.DataFrame: DataFrame with percentage changes of OHLC prices and volume.
    """
    tmp = df.copy()
    tmp["Volume"] = tmp["Volume"] / tmp["Close"]
    tmp[["Open", "High", "Low", "Close"]] = tmp[
        ["Open", "High", "Low", "Close"]
    ].pct_change()
    tmp["Open"] = np.log1p(tmp["Open"])
    tmp["High"] = np.log1p(tmp["High"])
    tmp["Low"] = np.log1p(tmp["Low"])
    tmp["Close"] = np.log1p(tmp["Close"])
    tmp[["Open", "High", "Low", "Close"]] = tmp[
        ["Open", "High", "Low", "Close"]
    ].cumsum()
    return tmp
