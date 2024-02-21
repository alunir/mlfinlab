import numpy as np
import pandas as pd


def compute_imbalance_bars(ohlcv, bucket_size=1e7):
    """
    Compute imbalance bars based on OHLCV data and VPIN concept.

    :param ohlcv: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
    :param bucket_size: Desired volume for each bucket (bar)
    :return: Imbalance bars (OHLC format)
    """

    ohlcv["Return"] = ohlcv["Close"].pct_change()
    ohlcv.dropna(inplace=True)

    ohlcv["SignedVolume"] = np.where(
        ohlcv["Return"] > 0,
        ohlcv["Volume"],
        np.where(ohlcv["Return"] < 0, -ohlcv["Volume"], 0),
    )

    bars = []
    running_volume = 0
    running_buy_volume = 0
    running_sell_volume = 0
    open_price = ohlcv.iloc[0]["Open"]

    index = []

    for idx, row in ohlcv.iterrows():
        running_volume += abs(row["SignedVolume"])
        running_buy_volume += row["SignedVolume"] if row["SignedVolume"] > 0 else 0
        running_sell_volume -= row["SignedVolume"] if row["SignedVolume"] < 0 else 0

        if running_volume >= bucket_size:
            close_price = row["Close"]
            high_price = max(ohlcv.loc[idx:idx]["High"])
            low_price = min(ohlcv.loc[idx:idx]["Low"])
            number_bars = sum(ohlcv.loc[idx:idx]["Number"])

            bars.append(
                {
                    "Open": open_price,
                    "High": high_price,
                    "Low": low_price,
                    "Close": close_price,
                    "Volume": running_volume,
                    "BuyVolume": np.log1p(running_buy_volume),
                    "SellVolume": np.log1p(running_sell_volume),
                    "Number": number_bars,
                }
            )

            index.append(idx)

            # Reset counters
            running_volume = 0
            running_buy_volume = 0
            running_sell_volume = 0
            open_price = close_price

    return pd.DataFrame(bars, index=index)
