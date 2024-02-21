import pandas as pd


def agg_ohlcv(df: pd.DataFrame, interval: int = 5) -> pd.DataFrame:
    """
    Aggregate OHLCV (Open, High, Low, Close, Volume) data based on a specified time interval.

    Args:
        df (pd.DataFrame): The input DataFrame containing OHLCV data.
        interval (int, optional): The time interval in minutes to aggregate the data. Defaults to 5.

    Returns:
        pd.DataFrame: The aggregated OHLCV DataFrame.

    """
    tmp = df.resample(f"{interval}min").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
            "Number": "sum",
        }
    )
    tmp["Open"] = tmp["Open"].ffill()
    tmp.fillna(
        {
            "High": tmp["Open"],
            "Low": tmp["Open"],
            "Close": tmp["Open"],
            "Volume": 0,
            "Number": 0,
        },
        inplace=True,
    )
    return tmp
