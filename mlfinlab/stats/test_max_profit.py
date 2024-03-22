import numpy as np
from mlfinlab.stats.max_profit import calculate_max_profit


def test_bull():
    mids = [100, 101, 103, 104, 106, 110]
    asks = [i + 1 for i in mids]
    bids = [i - 1 for i in mids]
    max_profit = calculate_max_profit(asks, bids, 0.01)
    assert np.isclose(
        max_profit.max_profit, 5.82
    ), f"MaxProfit should be 5.82, but got {max_profit.max_profit}"
    assert max_profit.buy_trades == [0]
    assert max_profit.sell_trades == [5]


def test_bear():
    mids = [110, 106, 104, 103, 101, 100]
    asks = [i + 1 for i in mids]
    bids = [i - 1 for i in mids]
    max_profit = calculate_max_profit(asks, bids, 0.01)
    assert np.isclose(
        max_profit.max_profit, 5.82
    ), f"MaxProfit should be 5.82, but got {max_profit.max_profit}"
    assert max_profit.buy_trades == [5]
    assert max_profit.sell_trades == [0]


def test_bull_then_bear():
    mids = [100, 101, 103, 104, 106, 110, 106, 104, 103, 101, 100]
    asks = [i + 1 for i in mids]
    bids = [i - 1 for i in mids]
    max_profit = calculate_max_profit(asks, bids, 0.01)
    assert np.isclose(
        max_profit.max_profit, 11.64
    ), f"MaxProfit should be 11.64, but got {max_profit.max_profit}"
    assert max_profit.buy_trades == [0, 10]
    assert max_profit.sell_trades == [5]


def test_bear_then_bull():
    mids = [110, 106, 104, 103, 101, 100, 101, 103, 104, 106, 110]
    asks = [i + 1 for i in mids]
    bids = [i - 1 for i in mids]
    max_profit = calculate_max_profit(asks, bids, 0.01)
    assert np.isclose(
        max_profit.max_profit, 11.64
    ), f"MaxProfit should be 11.64, but got {max_profit.max_profit}"
    assert max_profit.buy_trades == [5]
    assert max_profit.sell_trades == [0, 10]
