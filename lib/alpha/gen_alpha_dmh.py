#%%
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from pytz import NonExistentTimeError
sys.path.append(os.path.dirname(os.getcwd()))
from lib.evaluate.Detrendor import detrend_series_isotonic

class DataLoader:
    def __init__(self, path: str = "/home/sida/YIPENG/Task2_FactorTiming/data/PredictorLSretWide.csv") -> None:    
        self._df = pd.read_csv(path).sort_values(by = "date")
        self._calendar = list(self._df["date"])

    def get(self, col: str, date: str, backward: int, skip_last: int = 1) -> pd.Series:
        """
            Skip_last: avoid forward looking. Ignore last data.
        """
        end_idx = self._calendar.index(date)
        start_idx = max(end_idx - backward, 0)
        rets = self._df[col][start_idx : end_idx - skip_last]
        days = self._calendar[start_idx : end_idx - skip_last]
        return pd.Series(rets.values, index = days, name = col)
    
    @property
    def calendar(self):
        return self._calendar

class BackTestor():
    def __init__(self, move_df: pd.DataFrame, ret_col: str) -> None:
        self.move_df = move_df
        self.ret_col = ret_col
        self.current_pnl = 0
        self.buy_time_list, self.sell_time_list = [], []
        self.refresh()
    
    def refresh(self):
        self.holding = 0
        self.buy_time = 0
        self.sell_time = 0
    
    def buy(self, buy_time, amount: int = 1):
        self.holding += amount
        self.buy_time = buy_time
        self.buy_time_list.append(buy_time)
    
    def sell(self, sell_time, amount: int = 1):
        if self.holding <= 0:
            return
        self.holding -= amount
        self.sell_time = sell_time
        self.sell_time_list.append(sell_time)

        self.current_pnl += self.calc_return()
        self.refresh()

    def clear(self):
        """Sell all holding at last day"""
        if self.holding == 0:
            return
        self.sell_time = self.move_df.index[-1]
        self.sell_time_list.append(self.sell_time)

        self.current_pnl += self.calc_return()
    
    def calc_return(self) -> float:
        assert self.buy_time < self.sell_time, "Sell time should be later than buy time"
        return self.move_df[self.buy_time : self.sell_time][self.ret_col][:-1].sum()

    def run(self):
        for idx, row in self.move_df.iterrows():
            if row["change"] == 0:
                pass
            elif row["change"] > 0:
                self.buy(idx, 1)
            elif row["change"] < 0:
                self.sell(idx, 1)
        self.clear()
        return self.current_pnl
    
    # This is the viualization part
    def plot_performance(self):
        # Create a DataFrame to hold buy and sell signals along with returns
        signals_df = pd.DataFrame(index=self.move_df.index)
        signals_df['Signal'] = 0.0
        signals_df['Price'] = self.move_df[self.ret_col]
        
        # Mark buy signals with 1 and sell signals with -1
        signals_df.loc[self.buy_time_list, 'Signal'] = 1.0
        signals_df.loc[self.sell_time_list, 'Signal'] = -1.0
        
        # Initialize the plot figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot the price series
        ax.plot(signals_df['Price'], label='Price', color='blue', lw=2)
        
        # Plot the buy signals
        ax.plot(signals_df[signals_df['Signal'] == 1.0].index, 
        signals_df['Price'][signals_df['Signal'] == 1.0],
        '^', markersize=10, color='g', lw=0, label='Buy Signal')
        
        # Plot the sell signals
        ax.plot(signals_df[signals_df['Signal'] == -1.0].index, 
        signals_df['Price'][signals_df['Signal'] == -1.0],
        'v', markersize=10, color='r', lw=0, label='Sell Signal')
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title('Trading Strategy Performance')
        
        # Add a legend
        ax.legend(loc='best')
        
        # Show the plot
        plt.show()
    
class PolicyGenerator():
    def __init__(self, col: str) -> None:
        self.col = col
        self.loader = DataLoader()

    def load_residuals(self, end_date: str, backward: int):
        data = self.loader.get(self.col, end_date, backward)
        resid = pd.DataFrame(
            detrend_series_isotonic(data, "detrend")
        )  # After detrending
        df = pd.merge(data, resid, left_index=True, right_index=True)
        self.train_df = df.head(len(df) - 6)
        self.test_df = df.tail(6)
        self.detrend_col = f"{self.col}_isotonic_detrend"

        return df

    @property
    def std(self):
        return self.train_df[self.detrend_col].std()

    @property
    def mean(self):
        return self.train_df[self.detrend_col].mean()
    
    def gen_movement_return(self, buy_bar: float, sell_bar: float = 0) -> BackTestor:
        train_df = self.train_df.copy()
        train_df["status"] = train_df[self.detrend_col].apply(
            lambda x: 1 if x < buy_bar else (-1 if x > sell_bar else 0)
        )
        train_df["change"] = train_df["status"].diff()
        train_df.at[train_df.index[0], "change"] = train_df["status"][0]
        self.move_df = train_df
        bt = BackTestor(train_df, self.col)

        return bt

    def grid_search(self, a_range, b_range) -> pd.DataFrame:
        res = []
        for a in a_range:
            for b in b_range:
                lower_bound = (a + b) * (-1)
                upper_bound = 0
                if lower_bound >= upper_bound:
                    continue

                bt = self.gen_movement_return(lower_bound, upper_bound)
                bt.run()
                backtestor.plot_performance()
                res.append(
                    {
                        "a": a,
                        "b": b,
                        "return": bt.current_pnl,
                        "buy_time": bt.buy_time_list,
                        "sell_time": bt.sell_time_list,
                    }
                )
        res = pd.DataFrame(res)
        res.sort_values(by="return", ascending=False, inplace=True)

        return res

col = "AM"
a, b = -0, 0
pg = PolicyGenerator(col)
pg.load_residuals("2019-12-31", 120)
res = pg.grid_search(np.linspace(-3 * pg.mean, 3 * pg.mean, 20),
                     np.linspace(-3 * pg.std, 3 * pg.std, 20))
res
# %%

# %%
