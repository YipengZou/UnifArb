import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from typing import Union, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from ..helper.DataLoader import DataLoader
from pykalman import KalmanFilter

class Detrendor:
    def __init__(self, col: str, window: int = 120, step: int = 3,
                 d_method = "istonic",
                 data_path = "/home/sida/YIPENG/RA_Tasks/UnifiedArbitrage/metadata/PredictorLSretWide.csv",
                 start_date: int = 20000101,
                 end_date: int = 20240101,) -> None:
        self._col = col
        self._window = window
        self._step = step
        self._data_path = data_path
        self._start_date = str(start_date)
        self._end_date = str(end_date)
        
        self.d_method = d_method
    
    @property
    def group_data(self):
        if not hasattr(self, "_group_data"):
            self._group_data = self.create_rolling_datasets(
                self.data, self._window, self._step
            )
        
        return self._group_data

    @property
    def preds(self) -> pd.Series:
        """Results of rolling detrending."""
        if not hasattr(self, "_preds"):
            self.batch_detrend()
        
        return self._preds # type: ignore
    
    @property
    def data(self):
        """Original data."""
        if not hasattr(self, "_data"):
            d = DataLoader(path = self._data_path)
            col_data = d.get_full_col(self._col)
            self._data = col_data[
                col_data.first_valid_index() : col_data.last_valid_index()
            ]
            s_time = pd.to_datetime(self._start_date).strftime("%Y-%m-%d")
            e_time = pd.to_datetime(self._end_date).strftime("%Y-%m-%d")
            self._data = self._data[s_time : e_time]

        return self._data
    
    @property
    def col(self):
        return self._col

    @staticmethod
    def create_rolling_datasets(series: pd.Series,
                                window_size: int = 120, step_size = 6):
        """
            Create rolling datasets for training
        """
        rolling_datasets = []
        assert len(series) >= window_size, "Series is too short."
        for i in range(0, len(series) - window_size + 1, step_size):
            window = series.iloc[i : i + window_size]
            rolling_datasets.append(window)

        return rolling_datasets
    
    @property
    def result(self) -> pd.DataFrame:
        df: pd.DataFrame = pd.concat([self.preds, self.data.cumsum()], axis = 1).sort_index()
        df = df.loc[df.first_valid_index() : df.last_valid_index()]
        df.columns = [self.detrend_col, self.col]

        return df
    
    def batch_detrend(self) -> pd.Series:
        """
            Use a part of data training, predict the future periods.
        """
        preds = []
        for data in self.group_data:
            data: pd.Series
            train_size = len(data) - self._step
            train_data = data.head(train_size)
            test_data = data.cumsum().tail(self._step)

            if self.d_method == "isotonic":
                model = detrend_series_isotonic(train_data, "model")
                pred = model.predict(list(range(train_size, len(data))))
            elif self.d_method == "kf":
                model, mean, cov = detrend_series_kalman(train_data, "model")
                pred = model.filter_update(mean, cov, list(range(train_size, len(data))))[0]
            pred = pd.Series(pred, index = test_data.index)
            pred = test_data - pred  # Detrend.
            preds.append(pred)

        self._preds = pd.concat(preds)
        self.detrend_col = f"{self._col}_{self.d_method}_detrend_p"
        self._preds.name = self.detrend_col

        return self.preds
    
    def full_detrend(self):
        """
            Use all data to train, predict the future periods.
        """
        if self.d_method == "isotonic":
            pred: pd.Series = detrend_series_isotonic(self.data, "detrend") # type: ignore
        elif self.d_method == "kf":
            pred: pd.Series = detrend_series_kalman(self.data, "detrend")
        pred.name = f"{self._col}_{self.d_method}_detrend_insample"

        return pred
    
    def plot_detrend(self):
        sns.set_style("darkgrid")
        plt.figure(figsize = (24, 12))
        self.det_df = pd.concat([
            self.preds, self.full_detrend(), self.data.cumsum()
        ], axis = 1).sort_index()
        self.det_df.index = self.det_df.index.map(pd.to_datetime)
        for col in self.det_df:
            sns.lineplot(self.det_df[col], label = col)
        plt.legend()

        return plt.gcf()
    
"""Detrend Cell."""
def detrend_series_ma(
        col: pd.Series, p: int = 12, 
        return_type: str = "detrend",
) -> pd.Series:
    """
        1. For a given factor return, drop the data until the first valid factor return
        2. Fill the following NaN with 0 and calculate the cumulative return
        3. Detrend the cumulative return with moving average
        4. Concat the NaN and detrended cumulative return, return.
    """
    col_use = col[col.first_valid_index() : col.last_valid_index()]
    
    y = col_use.fillna(0).cumsum()
    y_ = y.rolling(p, min_periods = 2).mean().\
        fillna(0).values

    if return_type == "detrend":
        return pd.Series(y - y_, index = col_use.index, 
                         name = f"{col.name}_isotonic_detrend")
    elif return_type == "predict":
        return pd.Series(y_, index = col_use.index, 
                    name = f"{col.name}_isotonic_predict")
    else:
        raise ValueError("return_type must be detrend or predict")

def detrend_series_linear(
        col: pd.Series,
        return_type: str = "detrend",
) -> pd.Series:
    """
        1. For a given factor return, drop the data until the first valid factor return
        2. Fill the following NaN with 0 and calculate the cumulative return
        3. Detrend the cumulative return with linear regression
        4. Concat the NaN and detrended cumulative return.
    """
    col_use = col[col.first_valid_index() : col.last_valid_index()]

    x = np.arange(len(col_use))
    x = np.array([np.ones(len(x)), x]).T
    y = col_use.fillna(0).cumsum()
    reg = LinearRegression().fit(x, y)
    y_ = reg.predict(x)

    if return_type == "detrend":
        return pd.Series(y - y_, index = col_use.index, 
                         name = f"{col.name}_linear_detrend")
    elif return_type == "predict":
        return pd.Series(y_, index = col_use.index, 
                    name = f"{col.name}_linear_predict")
    else:
        raise ValueError("return_type must be detrend or predict")

def detrend_series_isotonic(
        col: pd.Series,   # DO NOT USE CUMSUM
        return_type: str = "detrend",
) -> Union[pd.Series, Callable]:
    """
        col: a series of factor returns
    """
    col_use = col[col.first_valid_index() : col.last_valid_index()]
    x = np.arange(len(col_use))
    y = col_use.fillna(0).cumsum().values
    ir = IsotonicRegression(out_of_bounds = "clip", increasing = "auto").\
        fit(x, y) # type: ignore
    y_ = ir.predict(x) 

    if return_type == "detrend":
        return pd.Series(y - y_, index = col_use.index, 
                    name = f"{col.name}_isotonic_detrend")
    elif return_type == "predict":
        return pd.Series(y_, index = col_use.index, 
                    name = f"{col.name}_isotonic_predict")
    elif return_type == "model":
        return ir # type: ignore
    else:
        raise ValueError("return_type must be detrend or predict")


def detrend_series_kalman(
        col: pd.Series,   # DO NOT USE CUMSUM
        return_type: str = "detrend",
) -> Union[pd.Series, Callable, KalmanFilter]:
    """
        col: a series of factor returns
    """
    col_use = col[col.first_valid_index() : col.last_valid_index()]
    x = np.arange(len(col_use))
    y = col_use.fillna(0).cumsum().values
    
    # Define the Kalman Filter
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    state_means, covs = kf.filter(y)
    
    if return_type == "detrend":
        return pd.Series(y - state_means.flatten(), index=col_use.index, 
                    name=f"{col.name}_kalman_detrend")
    elif return_type == "predict":
        return pd.Series(state_means.flatten(), index=col_use.index, 
                    name=f"{col.name}_kalman_predict")
    elif return_type == "model":
        return kf, state_means[-1], covs[-1] # type: ignore
    else:
        raise ValueError("return_type must be detrend, predict, or model")
