from .Detrendor import Detrendor
from ..simulator.BackTestor import BackTestor
import pandas as pd
import numpy as np

class PolicyGenerator:
    def __init__(self, dt: Detrendor, window: int = 10) -> None:
        self._col = dt.col
        self._dt = dt
        self._window = window

        self.dcol = dt.detrend_col
        self._boll_p = 20
    
    @property
    def df(self) -> pd.DataFrame:
        if not hasattr(self, "_df"):
            self._df = self._dt.result.copy()
        
        return self._df # type: ignore
    
    @property
    def result(self):
        return self.res
    
    @property
    def best_bt(self) -> BackTestor:
        return self.gen_movement_return(
            *self.result.iloc[0][["a", "b"]].values
        )

    def gen_boll_bond(self, df):
        df["rolling_mean"] = df[self._col].\
            rolling(window = self._boll_p, min_periods = 3, closed = "left").mean()
        df["rolling_std"] = df[self._col].\
            rolling(window = self._boll_p, min_periods = 3, closed = "left").std()
        
        df["boll_u"] = df["rolling_mean"] + 2 * df["rolling_std"]
        df["boll_l"] = df["rolling_mean"] - 2 * df["rolling_std"]
        
        return df

    def gen_movement_return(self, a: float, b: float = 0) -> BackTestor:
        df = self.df.copy()
        df["det_mean"] = df[self.dcol].\
            rolling(window = self._window, min_periods = 3, closed = "left").mean()
        df["det_vol"] = df[self.dcol].\
            rolling(window = self._window, min_periods = 3, closed = "left").std()

        # df["u_bound"] = a * df["det_mean"] + b * df["det_vol"]
        df["u_bound"] = a * df["det_mean"] + b * df["det_vol"]
        df["l_bound"] = a * df["det_mean"] - b * df["det_vol"]
        df["status"] = np.where(df[self.dcol] < df["l_bound"], 1,
                          np.where(df[self.dcol] > df["u_bound"], -1, 0))
        df["change"] = df["status"]
        df = self.gen_boll_bond(df)

        self.move_df = df
        bt = BackTestor(df, self._col)

        return bt

    def grid_search(self, a_range, b_range) -> pd.DataFrame:
        res = []
        for a in a_range:
            for b in b_range:
                bt = self.gen_movement_return(a, b)
                bt.run()
                res.append(
                    {
                        "a": round(a, 2),
                        "b": round(b, 2),
                        "start_date": self.df.index[0],
                        "end_date": self.df.index[-1],
                        "return": round(bt.current_pnl, 4)*100,
                        "hold_return": round(self.df[self._col].sum(), 4)*100,
                        "b_date": bt.buy_time_list,
                        "s_date": bt.sell_time_list,
                        "pnl_list": bt.pnl_list,
                        "stop_loss_date": bt.stop_loss_list,
                        "winning_rate": round((np.array(bt.pnl_list)>0).mean(), 4)*100,
                        "avg_hold_p": round(np.array(bt.hold_p_list).mean(), 2),
                        "avg_return": round(np.array(bt.pnl_list).mean(), 4)*100,
                    }
                )
        res = pd.DataFrame(res)
        res.sort_values(by="return", ascending=False, inplace=True)
        res.reset_index(drop=True, inplace=True)
        self.res = res
        return res.head(1)
