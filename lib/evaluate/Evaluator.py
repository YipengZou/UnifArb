import numpy as np
import pandas as pd
from typing import Callable, Tuple
from statsmodels.tsa.ar_model import AutoReg

class ArbitrageEvaluator:
    def __init__(self) -> None:
        pass

    def get_excess_kurt(self, col: pd.Series) -> float:
        """Give a series of factor returns, return kurtosis - 3"""
        col_use = col[col.first_valid_index():].fillna(0)
        return ((col_use - col_use.mean())**4).mean() \
            / ((col_use - col_use.mean())**2).mean()**2 - 3

    def get_noise_to_signal_ratio(self, resid: pd.Series, y_true: pd.Series) -> float:
        """Calculate Var(epsilon) / Var(y)"""
        return np.var(resid) / np.var(y_true)

    def get_reverting_speed(self, rho: float) -> float:
        """Given the AR(1) coefficient, calculate the reverting speed"""
        return 1 - np.abs(rho)

    def get_y_std(self, col: pd.Series) -> float:
        """Given a series of factor returns, return the std"""
        col_use = col[col.first_valid_index():].fillna(0)
        return np.std(col_use)

    def fit_ts_model(
            self, col: pd.Series, lag: int = 1
    ) -> Tuple[int, float, Callable]:
        """
            1. For a given factor return, drop the NaN until the first valid factor return
            2. Fit AR(n) models for n = 1, 2, ..., 24, compare BIC values
            3. Choose the lag with the minimum BIC value
            4. The root is corresponding to the lag with the minimum BIC value
            5. If the root is larger than 1, re-fit the model using AR(1)
        """
        col_use = col[col.first_valid_index():].fillna(0).values
        bic_list, root_list, model_list = [], [], []
        max_lag = 24
        for lag in range(1, max_lag):
            model = AutoReg(col_use, lags = lag).fit()
            bic_list.append(model._results.bic)
            root_list.append(max(abs(model._results.roots)))
            model_list.append(model)

        lag_use = np.argmin(bic_list) + 1
        root = root_list[np.argmin(bic_list)]
        model = model_list[np.argmin(bic_list)]

        if root >= 1:  # non-stationary model, re-fit it using AR(1)
            lag_use = 1
            model = AutoReg(col_use, lags = lag_use).fit()
            root = np.abs(model.params[1])
        
        return lag_use, root, model

    def eval_arbitrage_award(
            self, col: pd.Series, 
            kurt_hate: float = 0.1, mrs_factor: float = 1.0,
    ) -> Tuple[float]:
        lag_use, root, model = self.fit_ts_model(col)
        mrs_value = self.get_reverting_speed(root)
        nsr_value = self.get_noise_to_signal_ratio(
            model.resid, col[-model.nobs:]
        )  # To keep obeservations same length
        std_value = self.get_y_std(col)
        excess_kurt = self.get_excess_kurt(col)
        
        period_reward = mrs_value * std_value * nsr_value
        utility = period_reward - kurt_hate * excess_kurt + mrs_factor * mrs_value

        return utility, mrs_value, std_value, nsr_value,\
            excess_kurt, lag_use, root