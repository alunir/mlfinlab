from dataclasses import dataclass


@dataclass
class MaxProfit:
    max_profit: float
    buy_trades: list[int]
    sell_trades: list[int]


def consolidate_sequence(seq):
    # Sort the sequence to ensure the consecutive numbers can be found easily.
    sorted_seq = sorted(seq)
    result = []
    current_sequence = [sorted_seq[0]]

    for number in sorted_seq[1:]:
        if number - current_sequence[-1] == 1:
            # If the current number is consecutive, add it to the current sequence.
            current_sequence.append(number)
        else:
            # If not consecutive, add the highest number of the current sequence to the result
            # and start a new sequence.
            result.append(current_sequence[-1])
            current_sequence = [number]

    # Add the highest number of the last sequence to the result.
    result.append(current_sequence[-1])
    return result


def calculate_max_profit(
    asks: list[int], bids: list[int], fee_ratio: float, verbose: bool = False
) -> MaxProfit:
    n = len(asks)
    sell_max_sell, sell_max_buy = float("-inf"), 0
    buy_max_sell, buy_max_buy = 0, float("-inf")

    sell_trades = []
    buy_trades = []
    for i in range(n):
        if sell_max_sell < sell_max_buy + bids[i] * (1.0 - 2 * fee_ratio):
            if verbose:
                print(f"sell sell? {i}")
            sell_max_sell = sell_max_buy + bids[i] * (1.0 - 2 * fee_ratio)
            sell_trades += [i] if i > 0 else [0]
        if sell_max_buy < sell_max_sell - asks[i]:
            if verbose:
                print(f"sell buy? {i}")
            sell_max_buy = sell_max_sell - asks[i]

        if buy_max_sell < buy_max_buy + bids[i] * (1.0 - 2 * fee_ratio):
            if verbose:
                print(f"buy sell? {i}")
            buy_max_sell = buy_max_buy + bids[i] * (1.0 - 2 * fee_ratio)
        if buy_max_buy < buy_max_sell - asks[i]:
            if verbose:
                print(f"buy buy? {i}")
            buy_max_buy = buy_max_sell - asks[i]
            buy_trades += [i] if i > 0 else [0]

    return MaxProfit(
        max_profit=sell_max_buy + buy_max_sell,
        buy_trades=consolidate_sequence(buy_trades),
        sell_trades=consolidate_sequence(sell_trades),
    )
