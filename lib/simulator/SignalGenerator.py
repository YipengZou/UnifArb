from collections import deque
import numpy as np
import pandas as pd
from typing import Tuple
from ..helper import ut as ut

class SignalGenerator:
    """
        A general signal class.
        Designed for generating different signals. No need to be restricted by certain form.
    """

    def __init__(self) -> None:
        self.bs_bound_info = {
            "index": [],
            "low_bound": [],
            "up_bound": [],
        }
        self.stp_bound_info = {
            "index": [],
            "low_bound": [],
        }
    
    def initialize(self, method: str = "standard",
                   bs_period: int = 10, track_return: int = 20,
                   n_boots: int = 10000, vol_period: int = 10,
                     **params):
        """
            Parameters:
            ----------
                - method: str
                    The method to generate signal. Should be one of ["standard", "hampel"]
                - bs_period: int
                    Look back bs_period of days to calc mean / var
                - track_return: int
                    Track the past track_return days return, use bootstrap to find distribution of current return
                - n_boots: int
                    Number of bootstrap
        """
        self.method = method
        self.bs_period = bs_period 
        self.track_return = track_return
        self.n_boots = n_boots
        self.log_return_list = deque(maxlen = self.track_return)  # A fixed length list. Record the log_return
        self.vol_period = vol_period


    @property
    def summary(self):
        return {"method": self.method}

    @property
    def result(self) -> pd.DataFrame:
        return {
            "bs_bound_info": pd.DataFrame(self.bs_bound_info).set_index("index"),
            "stp_bound_info": pd.DataFrame(self.stp_bound_info).set_index("index"),
        }
    
    @property
    def params(self) -> dict:
        if not hasattr(self, "_params"):
            self._params = {
                "low_b": -3.0, "up_b": 3.0,
                "n_sigma_b": 1.0, "n_sigma_s": 2.0
            }
        return self._params
    
    @params.setter
    def params(self, 
               params: dict = {
                   "low_b": -3.0, "up_b": 3.0,
                     "n_sigma_b": 1.0, "n_sigma_s": 2.0
               }):
        self._params = params

    @property
    def today(self) -> pd.Timestamp:
        return self._today
    
    @today.setter
    def today(self, today: pd.Timestamp):
        self._today = today
    
    @property
    def min_period(self) -> int:
        return max(self.bs_period, self.track_return)
    
    @property
    def top_bs_upbound(self) -> float:
        """Return the most recent upper bound"""
        return self.bs_bound_info["up_bound"][-1]
    
    @property
    def top_bs_lowbound(self) -> float:
        """Return the most recent lower bound"""
        return self.bs_bound_info["low_bound"][-1]
    
    @property
    def top_signal(self) -> float:
        """Return the most recent signal"""
        return self._top_signal

    def gen_bs_signal(self, signal: pd.Series,
                      method: str = "standard") -> int:
        """
            Generate buy/sell/hold signal. Need 4 parameters.
            Use past signal data to determine mu/sigma.

            Returns:
            -------
                - 1: Buy signal
                - 0: Hold signal
                - -1: Sell signal
        """
        if method == "standard":
            weights = ut.calc_distance_weight(signal)
            avg, std = ut.weighted_avg_and_std(signal, weights)
            bound_up = self.params["up_b"] + self.params["n_sigma_s"] * std
            bound_low = self.params["low_b"] - self.params["n_sigma_b"] * std

        elif method == "bootstrap":
            (bound_low, bound_up), _ = \
                self.gen_boot_dist(signal, 0.01, 99.9, self.n_boots, 10)
            
        elif method == "hampel":
            median = signal.median()
            deviation = 1.4826 * (signal - median).abs().median()  # MAD, Median Absolute Deviation
            bound_up = median + self.params["n_sigma_s"] * deviation
            bound_low = median - self.params["n_sigma_b"] * deviation

        self._top_signal = signal.iloc[-1]

        self.bs_bound_info["index"].append(self.today)
        self.bs_bound_info["low_bound"].append(bound_low)
        self.bs_bound_info["up_bound"].append(bound_up)

        if self.top_signal < self.top_bs_lowbound:
            return 1  # Long signal
        elif self.top_signal > self.top_bs_upbound:
            return -1  # Short signal
        
        return 0  # Hold signal
    
    @staticmethod
    def gen_boot_dist(
            data: pd.Series, 
            low_percentile: float = 0.01, 
            high_percentile: float = 99.9, 
            n_boot: int = 1000, 
            group_size: int = 10,
    ) -> Tuple[Tuple[float, float], np.array]:
        """
            Use bootstrap method to find the distribution of recorded data. Return two percetiles.\n
            Parameters:
            ----------
                - low_percentile: float. 
                    The low percentile of the distribution. Default: 0.01
                - high_percentile: float. 
                    The high percentile of the distribution. Default: 99.9
                - n_boot: int. 
                    Number of bootstrap. Default: 1000
                - group_size: int. 
                    The size of each group. Default: 10\n

            Output:
            ------
                - (low_bound, high_bound): Tuple[float, float]
                    The low and high bound of the distribution of \mu
                - boot_ret_list: np.array
                    The list of all bootstrapped \mu

            Theory:
            -------
                - We do not know the distribution of \mu. We only have bar_x
                - We know \sigma = bar_x - \mu. -> If we know the disrtibution of sigma -> know the distribution of \mu
                - Bootstrap: 
                    - Everytime we sample a small group and calculate bar_x^* (intermediate variable)
                    - Record the difference between bar_x^* and bar_x as \sigma^*
                    - Repeat the process for n times. Record the distribution of \sigma^*
                    - Estimate the distribution of \mu by [bar_x - high_percentile(\sigma^*), bar_x - low_percentile(\sigma^*)]
                        - This is because \mu = bar_x - \sigma
            //TODO: 试一下t-bootstrap
        """
        boot_ret_list = np.random.choice(data, size=(n_boot, group_size), replace=True)
        sample_mean = np.array(data).mean()
        errors = boot_ret_list.mean(axis=1) - sample_mean
        error_l, error_h = np.percentile(errors, [low_percentile, high_percentile])
        return (sample_mean - error_h, sample_mean - error_l), boot_ret_list
    
    def gen_stop_loss_signal(self, hold_ret: float, hold_period: int = 1, 
                             log_return: bool = True) -> int:
        """
            Parameters:
            ----------
                - hold_ret: float
                    The return of holding the position for hold_period days.
                        P_{t+k} / P_t
                - hold_period: int
                    The number of days holding the position.
                - log_return: bool
                    Whether the return needs to take log.
        """
        if log_return:
            hold_ret = np.log(hold_ret)
        avg_ret = hold_ret / max(hold_period, 1)  # Period average return
        (stop_loss_ret, upper_b), _ = \
            self.gen_boot_dist(0.01, 99.9, self.n_boots, 10)
        
        self.stp_bound_info["index"].append(self.today)
        self.stp_bound_info["low_bound"].append(stop_loss_ret)

        if len(self.stp_bound_info["low_bound"]) > 1:
            if avg_ret < self.stp_bound_info["low_bound"][-2]:  
                # The bound info bootstrap does not contain current day
                return 1
            
        return 0  # hold
    
    def gen_stop_earning_signal(self):
        #// TODO: Add stop earning signal
        return