import numpy as np
import pandas as pd
from dataclasses import dataclass
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


"""
Univariate Local Linear Trend Model
"""


class LocalLinearTrend(sm.tsa.statespace.MLEModel):
    def __init__(self, endog):
        # Model order
        k_states = k_posdef = 2

        # Initialize the statespace
        super(LocalLinearTrend, self).__init__(
            endog,
            k_states=k_states,
            k_posdef=k_posdef,
            initialization="approximate_diffuse",
            loglikelihood_burn=k_states,
        )

        # Initialize the matrices
        self.ssm["design"] = np.array([1, 0])
        self.ssm["transition"] = np.array([[1, 1], [0, 1]])
        self.ssm["selection"] = np.eye(k_states)

        # Cache some indices
        self._state_cov_idx = ("state_cov",) + np.diag_indices(k_posdef)

    @property
    def param_names(self):
        return ["sigma2.measurement", "sigma2.level", "sigma2.trend"]

    @property
    def start_params(self):
        return [np.std(self.endog)] * 3

    def transform_params(self, unconstrained):
        return unconstrained**2

    def untransform_params(self, constrained):
        return constrained**0.5

    def update(self, params, *args, **kwargs):
        params = super(LocalLinearTrend, self).update(params, *args, **kwargs)

        # Observation covariance
        self.ssm["obs_cov", 0, 0] = params[0]

        # State covariance
        self.ssm[self._state_cov_idx] = params[1:]


@dataclass
class LocalLinearTrendResult:
    inverse_transform: np.array
    y: pd.Series
    predicted_mean: pd.Series
    predicted_conf: pd.DataFrame
    forecast_mean: pd.Series
    forecast_conf: pd.DataFrame

    def plot(self, ylim=None):
        fig, ax = plt.subplots(figsize=(10, 4))

        # Plot the results
        self.y.plot(ax=ax, style="k.", label="Observations")
        self.predicted_mean.plot(ax=ax, label="One-step-ahead Prediction")
        predict_index = np.arange(len(self.predicted_conf))
        ax.fill_between(
            predict_index[2:],
            self.predicted_conf.iloc[2:, 0],
            self.predicted_conf.iloc[2:, 1],
            alpha=0.3,
        )

        self.forecast_mean.plot(ax=ax, style="r", label="Forecast")
        forecast_index = np.arange(
            len(self.predicted_conf), len(self.predicted_conf) + len(self.forecast_conf)
        )
        ax.fill_between(
            forecast_index,
            self.forecast_conf.iloc[:, 0],
            self.forecast_conf.iloc[:, 1],
            alpha=0.3,
        )

        # Cleanup the image
        if ylim:
            ax.set_ylim(ylim)
        ax.legend(loc="lower left")


def llt_transform(
    s: pd.Series, scaler=StandardScaler, forecast=100, alpha=0.05
) -> LocalLinearTrendResult:
    _s = s.reset_index(drop=True)

    _scaler = scaler()
    y = pd.Series(
        _scaler.fit_transform(_s.values.reshape(-1, 1)).reshape(
            -1,
        ),
        index=_s.index,
    )

    # Setup the model
    mod = LocalLinearTrend(y)

    # Fit it using MLE (recall that we are fitting the three variance parameters)
    res = mod.fit(disp=False)
    # print(res.summary())

    # Perform prediction and forecasting
    predict = res.get_prediction()
    forecast = res.get_forecast(forecast)

    return LocalLinearTrendResult(
        inverse_transform=_scaler.inverse_transform,
        y=y,
        predicted_mean=predict.predicted_mean,
        predicted_conf=predict.conf_int(alpha=alpha),
        forecast_mean=forecast.predicted_mean,
        forecast_conf=forecast.conf_int(),
    )
