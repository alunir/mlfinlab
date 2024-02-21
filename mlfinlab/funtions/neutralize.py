import pandas as pd
import numpy as np


def neutralize_series(series: pd.Series, by: pd.Series, proportion=1.0):
    """
    Neutralizes a pandas series by removing the influence of another series.

    Parameters:
    series (pd.Series): The series to be neutralized.
    by (pd.Series): The series whose influence will be removed from the original series.
    proportion (float, optional): The proportion of the influence to be removed. Defaults to 1.0.

    Returns:
    pd.Series: The neutralized series.
    """
    scores = series.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)
    exposures = np.hstack(
        (exposures, np.array([np.mean(series)] * len(exposures)).reshape(-1, 1))
    )
    correction = proportion * (exposures.dot(np.linalg.lstsq(exposures, scores)[0]))
    corrected_scores = scores - correction
    neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    return neutralized
