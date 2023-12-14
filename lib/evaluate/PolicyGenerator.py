from .Detrendor import Detrendor
from ..simulator.BackTestor import BackTestor
import pandas as pd
import numpy as np
import optuna
from optuna.trial import Trial

class PolicyGenerator:
    def __init__(self, dt: Detrendor, window: int = 10) -> None:
        self._col = dt.col
        self._dt = dt
        self._window = window

        self.dcol = dt.detrend_col
        self._boll_p = 20
        self._best_criteria = np.inf
        self._best_res = {}
        self._best_bt: BackTestor
    
    @property
    def df(self) -> pd.DataFrame:
        if not hasattr(self, "_df"):
            self._df = self._dt.result.copy()
        
        return self._df # type: ignore
    
    @property
    def best_bt(self) -> BackTestor:
        return self._best_bt
    
    @property
    def best_result(self):
        return self._best_res

    def gen_stop_loss(self, df, stop_loss: float = 3):
        """布林带止损"""
        df["rolling_mean"] = df[self._col].\
            rolling(window = self._boll_p, min_periods = 3, closed = "left").mean()
        df["rolling_std"] = df[self._col].\
            rolling(window = self._boll_p, min_periods = 3, closed = "left").std()
        
        df["stop_earning_bar"] = df["rolling_mean"] + stop_loss * df["rolling_std"]
        df["stop_loss_bar"] = df["rolling_mean"] - stop_loss * df["rolling_std"]
        
        return df

    def get_decision_df(self, a: int, b: int = 0, c: float = 1,
                        stop_loss: float = 3, **kwargs,
                        ) -> pd.DataFrame:
        df = self.df.copy()
        df["det_mean"] = df[self.dcol].\
            rolling(window = self._window, min_periods = 3, closed = "left").mean()
        df["det_vol"] = df[self.dcol].\
            rolling(window = self._window, min_periods = 3, closed = "left").std()

        # df["u_bound"] = a * df["det_mean"] + b * df["det_vol"]
        # df["l_bound"] = a * df["det_mean"] - b * df["det_vol"]
        df["u_bound"] = a + c * df["det_vol"]
        df["l_bound"] = b - c * df["det_vol"]
        df["status"] = np.where(df[self.dcol] < df["l_bound"], 1,
                          np.where(df[self.dcol] > df["u_bound"], -1, 0))
        df["change"] = df["status"]
        df = self.gen_stop_loss(df, stop_loss = stop_loss)

        self.move_df = df

        return df

    def eval_bt_result(self, bt: BackTestor, params: dict) -> dict:
        """Evaluate the backtest result"""
        res = {
            "start_date": bt.move_df.index[0],
            "end_date": bt.move_df.index[-1],
            "return": round(bt.PnL, 4)*100,
            "hold_return": round(self.df[self._col].sum(), 4)*100,
            "b_date": bt.buy_time_list,
            "s_date": bt.sell_time_list,
            "pnl_list": bt.pnl_list,
            "stop_loss_date": bt.stop_loss_list,
            "stop_earning_date": bt.stop_earning_list,
            "winning_rate": round((np.array(bt.pnl_list) > 0).mean(), 4)*100,
            "avg_hold_p": round(np.array(bt.hold_p_list).mean(), 2),
            "avg_return": round(np.array(bt.pnl_list).mean(), 4),
        }
        res["ann_return"] = (1 + res["avg_return"]  / res["avg_hold_p"]) ** 250 - 1
        params.update(res)
        return params

    def grid_search(self, a_range, b_range) -> dict:
        for a in a_range:
            for b in b_range:
                decison_df = self.get_decision_df(a, b)
                bt = BackTestor(decison_df, self._col)
                bt.run()
                params = {
                    "a": round(a, 2),
                    "b": round(b, 2),
                }
                bt_res = self.eval_bt_result(bt, params)
                if bt_res["ann_return"] < self._best_criteria:
                    self._best_criteria = bt_res["ann_return"]
                    self._best_res = bt_res
                    self._best_bt = bt

        return self._best_res
    
    def bayes_objective(self, trial: Trial):
        param_space = {
            "a": trial.suggest_int("a", 3, 15),
            "b": trial.suggest_int("b", -15, -3),
            "c": trial.suggest_float("c", 0.5, 2),
            "d": trial.suggest_float("d", 0.5, 2),
            "stop_loss": trial.suggest_float("stop_loss", 4, 6),
            "max_earning": trial.suggest_categorical("max_earning", list(np.arange(0.03, 0.20, 0.01))),
        }
        decison_df = self.get_decision_df(**param_space)
        bt = BackTestor(decison_df, self._col, 
                        max_earning = param_space["max_earning"])
        bt.run()
        bt_res = self.eval_bt_result(bt, param_space)
        criteria = bt_res["avg_return"] * len(bt_res["pnl_list"])

        if criteria < self._best_criteria:
            self._best_criteria = criteria
            self._best_res = bt_res
            self._best_bt = bt

        return criteria
    
    def bayes_search(self, n_trials: int = 10):
        study = optuna.create_study(direction = "maximize")
        study.optimize(self.bayes_objective, n_trials = n_trials)
        return
