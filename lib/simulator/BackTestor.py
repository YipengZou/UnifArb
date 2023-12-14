import pandas as pd
import numpy as np

class BackTestor:
    def __init__(self, move_df: pd.DataFrame, 
                 ret_col: str, empty_last: bool = False,
                 max_earning: float = 0.1) -> None:
        self.move_df = move_df
        self.ret_col = ret_col  # factor cum return column
        self._today: str = move_df.index[0]

        self._min_hold_p = 10  # min holding period
        self._calm_p = 10  # After stop loss, do not open position for _calm_p days.
        self._max_holding = 1  # Number of holdings
        self._empty_last = empty_last  # Clear position in the last date.
        self._max_earning = max_earning  # Percentage of earning. Exceed -> sell.

        self._current_pnl = 0
        self.buy_time_list, self.sell_time_list = [], []
        self.stop_loss_list = []
        self.stop_earning_list = []
        self.pnl_list, self.hold_p_list = [], []
        self._calm_flag: bool = False

        self.refresh()
    
    @property
    def hold_return(self) -> float:
        """Return from buy date to today"""
        return self._hold_ret

    @property
    def sell_return(self) -> float:
        """
            Return from buy date to sell date
        """
        assert self.b_date < self.s_date, "Sell time should be later than buy time"
        series = self.move_df[self.b_date : self.s_date][self.ret_col]
        
        return series.iloc[-1] - series.iloc[0]
    
    @property
    def hold_p(self) -> int:
        """Current Holding Time"""
        return self._hold_p
    
    @property
    def position(self) -> int:
        """Current holding number"""
        return self._position

    @property
    def calm(self) -> bool:
        """Determine whether now we should not make any decision"""
        if not self._calm_flag:
            return False
        else:
            day_to_calm = self.get_days_gap(self._today, self.stop_loss_list[-1])
            if day_to_calm >= self._calm_p:
                self._calm_flag = False
                return False
            else:
                return True
    
    @property
    def PnL(self) -> float:
        """Current PnL"""
        return self._current_pnl
    
    def refresh(self):
        self._position = 0
        self._hold_p: int = 0
        self._hold_ret: float = 0
        self.b_date: str = ""
        self.s_date: str = ""
    
    def buy(self, amount: int = 1):
        if self.position >= self._max_holding:  # Full position.
            return
        self._position += amount
        self.b_date = self._today
        self.buy_time_list.append(self._today)
    
    def sell(self, amount: int = 1):
        if self.position <= 0:  # Empty position.
            return
        self._position -= amount
        self.s_date = self._today
        self.sell_time_list.append(self._today)

        self._current_pnl += self.sell_return
        self.pnl_list.append(self.sell_return)
        self.hold_p_list.append(self.hold_p)
        self.refresh()

    def clear(self):
        """Sell all holding at last day"""
        if self.position == 0:
            return
        self.s_date = self.move_df.index[-1]
        self.sell_time_list.append(self.s_date)

        self._current_pnl += self.sell_return

    def stop_loss(self, row: pd.Series):
        """Lower than stop_loss_bar -> sell"""
        if self.position <= 0 or self.hold_p < self._min_hold_p:
            return
        else:
            if row[self.ret_col] < row["stop_loss_bar"]:
                self.sell(1)
                self.stop_loss_list.append(self._today)
                self._calm_flag = True

    def stop_earning(self, row: pd.Series):
        """Greater than stop_earning_bar -> sell"""
        if self.position <= 0 or self.hold_p < self._min_hold_p:
            return
        else:
            if self.hold_return >= self._max_earning:
                self.sell(1)
                self.stop_earning_list.append(self._today)

            elif row[self.ret_col] > row["stop_earning_bar"]:
                self.sell(1)
                self.stop_earning_list.append(self._today)

    def update_holding(self):
        self._hold_p += 1

        hold_series = self.move_df[self.b_date : self._today][self.ret_col]
        self._hold_ret = hold_series.iloc[-1] - hold_series.iloc[0]

    def get_date_idx(self, today: str) -> int:
        if today not in self.move_df.index:
            return -1
        
        return self.move_df.index.get_loc(today)
    
    def get_days_gap(self, day1: str, day2: str) -> int:
        """Get the number of days between day1 and day2. day1: str, day2: str"""
        idx1 = self.get_date_idx(day1)
        idx2 = self.get_date_idx(day2)
        if idx1 == -1 or idx2 == -1:
            return 0
        return np.abs(idx2 - idx1)

    def run(self):
        for t_date, row in self.move_df.iterrows():
            self._today = str(t_date)
            if self.calm:  # During calm down period, donot make any decision.
                continue
            else:
                if row["change"] == 0:
                    pass
                elif row["change"] > 0:
                    self.buy(1)
                elif row["change"] < 0:
                    self.sell(1)

                if self.position > 0:  # Currently have position
                    self.update_holding()

                self.stop_loss(row)
                self.stop_earning(row)

        if self._empty_last:
            self.clear()

        return self.PnL