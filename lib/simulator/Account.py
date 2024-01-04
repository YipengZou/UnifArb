import numpy as np
import pandas as pd
from typing import List, Tuple

class Account:
    def __init__(self, item: str,
                 cash: float = 1000000.0, max_holding: float = 1.0, 
                 b_fee_rate: float = 0.001, s_fee_rate: float = 0.001,
                 **params):
        self._item = item
        self._cash = cash
        self._max_holding = max_holding
        self._params = params
        self._b_fee_rate = b_fee_rate
        self._s_fee_rate = s_fee_rate
        
        self._hold_amount = 0.0
        self._cur_price: float = 0.0
        self.b_time_list, self.b_price_list, self.b_price_amount = [], [], []
        self.s_time_list, self.s_price_list, self.s_price_amount = [], [], []
        self.b_fee_list, self.s_fee_list = [], []
    
    @property
    def cash(self) -> float:
        return self._cash
    
    @property
    def fee_rate(self) -> float:
        return self._b_fee_rate + self._s_fee_rate
    
    @property
    def hold_amount(self) -> int:
        """
            Hold how many shares
            NOTE: volume = amount * price
        """
        return self._hold_amount
    
    @property
    def hold_values(self) -> float:
        """Holding values"""
        return self.hold_amount * self._cur_price
    
    @property
    def value(self) -> float:
        return self.cash + self.hold_values
    
    @property
    def buyable_amount(self) -> int:
        """How many shares can be bought"""
        return np.floor(self.cash / self._cur_price)
    
    @property
    def sellable_amount(self) -> int:
        """How many shares can be sold"""
        return self.hold_amount
    
    def update(self, price: float):
        self._cur_price = price
    
    def buy(self, day: pd.Timestamp, price: float, amount: float = 1.0):
        """Logic of buying a share"""
        self._hold_amount += amount
        self._cash -= amount * price * (1 + self._b_fee_rate)

        self.b_time_list.append(day)
        self.b_price_list.append(price)
        self.b_price_amount.append(amount)
        self.b_fee_list.append(amount * price * self._b_fee_rate)

    def sell(self, day: pd.Timestamp, price: float, amount: float = 1.0):
        self._hold_amount -= amount
        self._cash += amount * price * (1 - self._s_fee_rate)
        self.s_time_list.append(day)
        self.s_price_list.append(price)
        self.s_price_amount.append(amount)
        self.s_fee_list.append(amount * price * self._s_fee_rate)