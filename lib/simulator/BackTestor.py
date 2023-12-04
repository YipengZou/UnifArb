import pandas as pd

class BackTestor:
    def __init__(self, move_df: pd.DataFrame, 
                 ret_col: str, empty_last: bool = False) -> None:
        self.move_df = move_df
        self.ret_col = ret_col
        self._today: str = move_df.index[0]
        self.min_hold_p = 10  # min holding period

        self.max_holding = 1
        self.current_pnl = 0

        self.buy_time_list, self.sell_time_list = [], []
        self.stop_loss_list = []
        self.pnl_list, self.hold_p_list = [], []
        self._empty_last = empty_last  # Clear position in the last date.
        self.refresh()
    
    def refresh(self):
        self.holding = 0
        self.hold_p: int = 0
        self.b_date: str = ""
        self.s_date: str = ""
    
    def buy(self, amount: int = 1):
        if self.holding >= self.max_holding:  # Full position.
            return
        self.holding += amount
        self.b_date = self._today
        self.buy_time_list.append(self._today)
    
    def sell(self, amount: int = 1):
        if self.holding <= 0:  # Empty position.
            return
        self.holding -= amount
        self.s_date = self._today
        self.sell_time_list.append(self._today)

        self.current_pnl += self.calc_return()
        self.pnl_list.append(self.calc_return())
        self.hold_p_list.append(self.hold_p)
        self.refresh()

    def clear(self):
        """Sell all holding at last day"""
        if self.holding == 0:
            return
        self.s_date = self.move_df.index[-1]
        self.sell_time_list.append(self.s_date)

        self.current_pnl += self.calc_return()

    def get_holding_ret(self, today: str):
        """
            today: str, format: %Y-%m-%d
            return: float
                cum return from buy date to today
        """
        if self.b_date == "" or today == self.b_date:
            return 0
        assert self.b_date < today, "Today should be later than buy date"

        series = self.move_df[self.b_date : today][self.ret_col]
        return series.iloc[-1] - series.iloc[0]
    
    def calc_return(self) -> float:
        """
            Logic: Buy at the date we found detrend value < l_bound.
            Sell at the date we found detrend value > u_bound. (May have the delay problem)
        """
        assert self.b_date < self.s_date, "Sell time should be later than buy time"
        series = self.move_df[self.b_date : self.s_date][self.ret_col]
        
        return series.iloc[-1] - series.iloc[0]

    def stop_loss(self, row: pd.Series):
        """Lower than boll_l -> sell"""
        if self.holding <= 0 or self.hold_p < self.min_hold_p:
            return
        else:
            if row[self.ret_col] < row["boll_l"]:
                self.sell(1)
                self.stop_loss_list.append(self._today)

    def run(self):
        for t_date, row in self.move_df.iterrows():
            self._today = t_date  # type: ignore
            if row["change"] == 0:
                pass
            elif row["change"] > 0:
                self.buy(1)
            elif row["change"] < 0:
                self.sell(1)

            if self.holding > 0:
                self.hold_p += 1
            self.stop_loss(row)
        if self._empty_last:
            self.clear()
        return self.current_pnl