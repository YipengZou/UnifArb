#%%
import pandas as pd
import numpy as np
from .SignalGenerator import SignalGenerator
from ..helper import ut as ut
from .Account import Account
from loguru import logger
from typing import Union

class BackTestor:
    def __init__(self, price: pd.Series, signal: pd.Series, 
                 sg: SignalGenerator):
        assert (price.index == signal.index).all(), "Index should be the same"
        
        self.price_col, self.signal_col = price.name, signal.name
        self._data = pd.concat([price, signal], axis = 1)
        self._sg = sg

        self.pnl_list, self.hold_p_list = pd.Series(), []
        self.stop_loss_list, self.stop_earning_list = [], []
        self.stop_loss_price_list, self.stop_earning_price_list = [], []
        self.low_vol_list, self.high_vol_list = pd.Series(), pd.Series()

        self.refresh()

    def initialize(
            self,
            bt_item: str = "GOLD",
            bt_cash: float = 1000000.0,
            act_b_fee: float = 0.001,
            act_s_fee: float = 0.001,
            bt_check_vol: bool = True,
    ):
        self.account = Account(
            item = bt_item,
            cash = bt_cash,
            b_fee_rate = act_b_fee,
            s_fee_rate = act_s_fee,
        )
        self.checkvol = bt_check_vol  # Check volatility or not
    
    def __repr__(self) -> str:
        return self.summary.__repr__()
    
    @property
    def data(self) -> pd.DataFrame:
        return self._data
    
    @property
    def signal(self) -> SignalGenerator:
        return self._sg
    
    @property
    def index(self) -> pd.Index:
        return self.data.index
    
    @property
    def valid_index(self) -> pd.Index:
        return self.data.index[self.signal.min_period :]

    @property
    def hold_period(self) -> int:
        if pd.isna(self.b_date):
            return 0
        else:
            return self.get_day_idx(self.today) \
                - self.get_day_idx(self.b_date) + 1

    @property
    def today(self) -> pd.Timestamp:
        return self._today
    
    @today.setter
    def today(self, today: pd.Timestamp):
        self._today = today

    @property
    def hold_return(self) -> float:
        if self.b_price == np.inf:
            return 1.0  # should be 1 instead of 0.
        cur_price = self.get_price(self.get_day_idx(self.today), None)
        return cur_price / self.b_price
    
    @property
    def summary(self) -> dict:
        return {
            "buy_time": self.account.b_time_list,
            "sell_time": self.account.s_time_list,
            "hold_period": self.hold_p_list,
            "b/s price": list(zip(self.account.b_price_list, self.account.s_price_list)),
            "b/s amount": list(zip(self.account.b_price_amount, self.account.s_price_amount)),
        }
    
    @property
    def bs_summary(self) -> pd.DataFrame:
        return self.signal.result["bs_bound_info"]
    
    @property
    def signal_summary(self) -> pd.DataFrame:
        return self.signal.result["stp_bound_info"]

    @property
    def stop_loss_summary(self) -> pd.DataFrame:
        return pd.DataFrame({
            "stop_loss_time": self.stop_loss_list,
            "stop_loss_price_list": self.stop_loss_price_list
        })
    
    @property
    def pnl(self) -> pd.Series:
        return self.pnl_list

    def get_signal(self, start: int, end: int) -> pd.Series:
        return self.data[self.signal_col].iloc[start : end]
    
    def get_price(self, start: int, end: int = None) -> Union[float, pd.Series]:
        if end is None:
            return self.data[self.price_col].iloc[start]
        else:
            return self.data[self.price_col].iloc[start : end]
    
    def get_day_idx(self, date: pd.Timestamp) -> int:
        try:
            return self.index.get_loc(date)
        except KeyError:
            logger.exception(f"{date} is not in the index")
            return -1
        
    def refresh(self):
        self._hold_ret: float = 0
        self.b_date: pd.Timestamp = pd.Timestamp(np.nan)
        self.s_date: pd.Timestamp = pd.Timestamp(np.nan)
        self.b_price: float = np.inf
        self.s_price: float = np.inf

        self.refresh_signal()
    
    def refresh_signal(self):
        self.ub_list = []
        self.lb_list = []
        self.buy_signal_period = 0
        self.sell_signal_period = 0

    def clear(self, price: float):
        """Sell all holding at last day"""
        if self.account.sellable_amount == 0:
            return
        self.sell(price, self.account.sellable_amount)
        
    def stop_earning(self, price: float):
        self.sell(price, self.account.sellable_amount)

    def stop_loss(self, price: float):
        # //TODO: 
        self.stop_loss_list.append(self.today)
        self.stop_loss_price_list.append(price)
        self.sell(price, self.account.sellable_amount)

    def buy(self, price: float, amount: float = 1.0):
        """Logic of buying a share"""

        self.b_date = self.today
        self.b_price = price
        self.account.buy(self.today, price, amount)

    def sell(self, price: float, amount: float = 1.0):
        self.s_date = self.today
        self.s_price = price

        self.hold_p_list.append(self.hold_period)

        self.account.sell(self.today, price, amount)
        self.refresh()

    def signal_check(self, mode: str = "buy") -> bool:
        """
            Buy:
                Current signal is positive or 
                    Current signal is 0 and previous signal is positive

            Sell:
                Current signal is negative or 
                    Current signal is 0 and previous signal is negative
        """
        if mode == "buy":
            return (self.bs_sig == 1) \
                or (self.bs_sig == 0 and self.prev_bs_sig == 1)
        
        elif mode == "sell":
            return (self.bs_sig == -1) \
                or (self.bs_sig == 0 and self.prev_bs_sig == -1)

    def account_check(self, mode: str = "buy") -> bool:
        if mode == "buy":
            return self.account.buyable_amount > 0
        elif mode == "sell":
            return self.account.sellable_amount > 0
        
    def buy_check(self) -> bool:
        if not (self.signal_check("buy") and self.account_check("buy")):
            return False
        
        low_sig = self.signal.top_signal
        self.buy_signal_period += 1
        self.lb_list.append(low_sig)
        if low_sig > min(self.lb_list):
            return True
        
        if low_sig > self.signal.top_bs_lowbound and self.buy_signal_period > 2:  # go up accross the low bound
            return True
        
        return False

    def sell_check(self) -> bool:
        if not (self.signal_check("sell") and self.account_check("sell")):
            return False
        
        up_sig = self.signal.top_signal
        self.sell_signal_period += 1
        self.ub_list.append(up_sig)

        if up_sig < max(self.ub_list):
            return True
        
        if up_sig < self.signal.top_bs_upbound and self.sell_signal_period > 2:  # go down accross the up bound
            # signal_period > 2 to avoid the case of accidentially cross the bound
            return True

        return False
    
    def vol_check(self) -> bool:
        if not self.checkvol:
            return True
        
        weights = ut.calc_distance_weight(self.price_use)
        avg, std = ut.weighted_avg_and_std(self.price_use, weights)

        if std / avg < self.account.fee_rate:
            return False
        
        return True

    def run(self) -> pd.DataFrame:
        for idx, (date, row) in enumerate(self.data.iterrows()):
            """Init process"""
            self.account.update(row[self.price_col])
            self.today = date
            self.signal.today = date

            if idx == 0: 
                cur_price, prev_price = row[self.price_col], np.nan
                self.prev_bs_sig, self.bs_sig = 0, 0
                continue
            cur_price, prev_price = row[self.price_col], cur_price
            cur_return = cur_price / prev_price  # period return
            self.signal.log_return_list.append(np.log(cur_return))

            if idx < self.signal.min_period: 
                continue

            self.signal_use = \
                self.get_signal(idx - self.signal.bs_period + 1, idx + 1)
            self.price_use: pd.Series = \
                self.get_price(idx - self.signal.vol_period + 1, idx + 1)

            if self.vol_check():
                self.high_vol_list[date] = cur_price
            else:
                self.low_vol_list[date] = cur_price

            self.prev_bs_sig, self.bs_sig = \
                self.bs_sig, self.signal.gen_bs_signal(
                    signal = self.signal_use, 
                    method = self.signal.method
                )

            if self.vol_check() and self.buy_check():
                self.buy(cur_price, self.account.buyable_amount)
            if self.sell_check():
                self.sell(cur_price, self.account.sellable_amount)

            if self.bs_sig == 0 and self.prev_bs_sig != 0:  # From buy/sell to hold
                self.refresh_signal()

            self.pnl_list[date] = self.account.value

            # stp_sig = self.signal.gen_stop_loss_signal(
            #     cur_return, 1, log_return = True
            # )  # Use current return and past distribution to calc stop loss signal
            
            # if self.hold_amount != 0 and self.hold_period > 3:
            #     if stp_sig == 1:
            #         self.stop_loss(cur_price)
            #     elif stp_sig == -1:
            #         self.stop_earning(cur_price)

        self.clear(cur_price)
        return self.summary


if __name__ == "__main__":
    import sys
    import pandas as pd
    PROJ_DIR = "/home/ubuntu/CryptArb"
    sys.path.append(PROJ_DIR)
    from lib.arbitrage.Detrendor import Detrendor
    from lib.simulator.SignalGenerator import SignalGenerator
    from lib.simulator.BackTestor import BackTestor
    from lib.plotter.plotter import BackTestorPlotter
    import lib.helper.ut as ut
    from tqdm import tqdm

    pre_step = 0
    df = pd.read_parquet(f"{PROJ_DIR}/data/bigdata/bn/5min/BTCUSDT_spot/BTCUSDT.parquet")
    df = df.loc[df["open_t_hk"] > ut.to_local_datetime("20231201")].reset_index(drop=True)

    d = Detrendor(df.tail(400), "vwap", "close_t_hk", 
                    cumsum = False, window = 120, step = 5)
    fitted_list, noise_list = [], []
    group_data = d.create_rolling_datasets(
        d.data, 
        window = d._window, 
        step = d._step, 
    )
    for data in tqdm(group_data, desc = "Detrending...", 
                    total = (len(d.data) - d._window) // d._step):
        data: pd.Series
        fitted, noise = d.outsample_detrend(
            train_data = data[pre_step : -d._step], 
            test_data = data[-d._step :], 
            method = "kf",
            return_fitted = False,
        )
        noise_list.append(noise)

    noise_result: pd.Series = pd.concat(noise_list)
    noise_result.name = d.noise_col
    d.noise = noise_result

    d.plot_result()
    sg = SignalGenerator()
    sg.initialize()
    sg.params = {"low_b": -100.0, "up_b": 100.0,
                "n_sigma_b": 1.0, "n_sigma_s": 2.0
                }
    bt = BackTestor(d.result["vwap"], d.result[d.noise_col], sg)
    summary = bt.run()
    bp = BackTestorPlotter(bt)
    bp.plot()