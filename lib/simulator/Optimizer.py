import pandas as pd
import numpy as np
import optuna
from optuna import Trial
from .SignalGenerator import SignalGenerator
from .BackTestor import BackTestor
from typing import Tuple

class Optimizer:
    def __init__(self, price: pd.Series, noise: pd.Series) -> None:
        self.price = price
        self._best_criteria = -np.inf
        self.noise = noise
        self._params = {
            "up_b": (3, 15),
            "low_b": (-15, -3),
            "n_sigma_b": (0.5, 2),
            "n_sigma_s": (0.5, 2),
        }

    def initialize(
            self,
            bt_item: str = "BTCUSDT",
            bt_cash: float = 1000000,
            bt_check_vol: bool = True,
            act_b_fee: float = 0.001,
            act_s_fee: float = 0.001,

            opt_up_b: Tuple[int, int] = (3, 15),
            opt_low_b: Tuple[int, int] = (-15, -3),
            opt_n_sigma_b: Tuple[float, float] = (0.5, 2),
            opt_n_sigma_s: Tuple[float, float] = (0.5, 2),

            sg_method: str = "standard",
            sg_bs_period: int = 10,
            sg_track_return: int = 20,
            sg_n_boots: int = 10000,
            **params,
    ):
        self._params = {
            "bt_item": bt_item,
            "bt_cash": bt_cash,
            "bt_check_vol": bt_check_vol,
            "opt_up_b": opt_up_b,
            "opt_low_b": opt_low_b,
            "opt_n_sigma_b": opt_n_sigma_b,
            "opt_n_sigma_s": opt_n_sigma_s,
            "act_b_fee": act_b_fee,
            "act_s_fee": act_s_fee,
            "sg_method": sg_method,
            "sg_bs_period": sg_bs_period,
            "sg_track_return": sg_track_return,
            "sg_n_boots": sg_n_boots,
            **params,
        }

    @property
    def bt(self) -> BackTestor:
        return self._best_bt
    
    @property
    def params(self):
        return self._params
    
    def bayes_objective(self, trial: Trial):
        param_space = {
            "up_b": trial.suggest_int("up_b", *self.params["opt_up_b"]),
            "low_b": trial.suggest_int("low_b", *self.params["opt_low_b"]),
            "n_sigma_b": trial.suggest_float("n_sigma_b", *self.params["opt_n_sigma_b"]),
            "n_sigma_s": trial.suggest_float("n_sigma_s", *self.params["opt_n_sigma_s"]),
        }
        sg = SignalGenerator()
        sg.initialize(
            method = self.params["sg_method"],
            bs_period = self.params["sg_bs_period"],
            track_return = self.params["sg_track_return"],
            n_boots = self.params["sg_n_boots"],
        )
        sg.params = param_space
        bt = BackTestor(self.price, self.noise, sg)
        bt.initialize(
            bt_item = self.params["bt_item"],
            bt_cash = self.params["bt_cash"],
            act_b_fee = self.params["act_b_fee"],
            act_s_fee = self.params["act_s_fee"],
            bt_check_vol = self.params["bt_check_vol"],
        )
        summary = bt.run()
        fee_bound: float = self.params["act_b_fee"] + self.params["act_s_fee"]
        win_rate = (pd.Series(summary["b/s price"])\
                    .apply(lambda x: x[1] / x[0] - 1) > fee_bound).mean()
        criteria = (bt.pnl.iloc[-1] / bt.pnl.iloc[0] - 1) * win_rate
        
        if criteria > self._best_criteria:
            self._best_criteria = criteria
            self._best_bt = bt

        return criteria
    
    def bayes_search(self, n_trials: int = 10):
        study = optuna.create_study(direction = "maximize")
        study.optimize(self.bayes_objective, n_trials = n_trials)