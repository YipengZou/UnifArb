#%%
#%%
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import warnings
import pprint
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from typing import Union, Tuple, Iterator
warnings.filterwarnings("ignore", category = UserWarning)
from pykalman import KalmanFilter
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Detrendor:
    def __init__(self, data: Union[pd.DataFrame, pl.DataFrame],
                 col: str, index_col: str = "time",
                 cumsum: bool = False, window: int = 120, step: int = 6,
                 ) -> None:
        """
            Parameters:
            ----------
            data: pd.DataFrame or pl.DataFrame
                Data to detrend. Need to contain `col` and `index_col` columns.
            col: str
                The column to detrend.
            index_col: str
                The index column. May be time or other index. Please seperate this index column into the dataframe
                Instead of using the default index.
            cumsum: bool
                Whether to cumsum the data. If the input is return, set it to True.
            
            Attributes:
            ----------
            window: int
                The size of a training set. Use this amount of data to train a small model and predict
            step: int
                The size of a test set. Predict for future `step` periods.
            method: str
                The method to detrend. Must be one of `isotonic`, `linear`, `kf`.
        """
        self._col = col  # Name of this column.
        self._index_col = index_col  # Name of index column.
        self._cumsum = cumsum  # Whether to cumsum the data. If the input is return, set it to True.

        if isinstance(data, pl.DataFrame):
            data = data.to_pandas()
        for c in [col, index_col]:
            assert c in data.columns, f"column: {c} not in data."
        self.check(data[col])
        
        if cumsum:
            self._data = self._data.cumsum()
        self._data: pd.DataFrame = data.copy()  # Use polars. Speed up the process.
    
        self._window: int = window  # Size of a training set
        self._step: int = step  # Size of a test set
        self.method: str

    def __repr__(self) -> str:
        return pprint.pformat(self.summary)
    
    @property
    def summary(self) -> dict:
        return {
            "col": self.col,
            "index_col": self.index,
            "window": self._window,
            "step": self._step,
            "method": self.method,
        }

    @property
    def data(self) -> pd.Series:
        """Original data. With index set. Only return the required index"""
        return self._data.set_index(self.index)[self.col]
    
    @property
    def col(self) -> str:
        return self._col
    
    @property
    def index(self) -> str:
        return self._index_col
    
    @property
    def noise_col(self) -> str:
        return f"{self.col}_{self.method}_detrend_p"
    
    @property
    def group_data(self) -> Iterator[pd.Series]:
        """Rolling datasets. Only use a part of data to train / predict."""
        return self.create_rolling_datasets(
                self.data, self._window, self._step
            )

    @property
    def noise(self) -> pd.Series:
        """Results of rolling detrending."""
        if not hasattr(self, "_noise"):
            raise ValueError("Please run the detrend method first.")
        
        return self._noise 
    
    @noise.setter
    def noise(self, value: pd.Series) -> None:
        self._noise = value
    
    @property
    def result(self) -> pd.DataFrame:
        if not hasattr(self, "_result"):
            self._result = pd.concat([self.data, self.noise], axis = 1).dropna()

        return self._result
    
    @staticmethod
    def check(series: pd.Series) -> bool:
        """Check whether the series is valid."""
        assert not series.isna().any(), f"Data check failed. Series contains NaN. \n{series.loc[series.isna()]}"
    
    @staticmethod
    def create_rolling_datasets(series: pd.Series, window: int = 120, 
                                step: int = 6, pre_step: int = 0) -> Iterator[pd.Series]:
        """
            Create rolling datasets for training.
            Use Iterator to save memory.
            Parameters:
            ----------
                - series: pd.Series
                    Data to create rolling datasets.
                - window: int
                    Size of a training set.
                - step: int
                    Size of a test set.
                - pre_step: int
                    The step to start rolling. If pre_step = 0, start from the first index.
                    If pre_step != 1: remain some extra data at the beginning.
        """
        assert len(series) >= window, "Series is too short."

        for i in range(pre_step, len(series) - window + 1, step):
            yield series.iloc[i - pre_step : i + window]
    
    def outsample_detrend(self, train_data: pd.Series, 
                          test_data: pd.Series, method: str, 
                          return_fitted: bool = False) -> Union[pd.Series, pd.Series]:
        """
            Use a part of data to train. Rolling predict for short future periods.
            Parameters:
            ----------
                - train_data: pd.Series
                    Data to train. Use this data to train a small model.
                - test_data: pd.Series
                    Data to predict. Predict for future `step` periods.
                - method: str
                    The method to detrend. Must be one of `isotonic`, `linear`, `kf`.

            Returns:
            -------
                - noise_: pd.Series
                    The detrended noise.
                - fitted: pd.Series
                    The fitted training data.
        """
        self.method = method
        train_size = len(train_data)
        test_size = len(test_data)
        batch_size = train_size + test_size

        if self.method == "isotonic":
            fitted, model = self.isotonic_detrend(
                data = train_data,
                return_type = "outsample",
                return_fitted = return_fitted,
            )
            pred = model.predict(list(range(batch_size - test_size, batch_size)))
            noise_ = test_data - pred

        elif self.method == "kf":
            res = []
            fitted, state, model = self.kalman_detrend(
                data = train_data, 
                return_type = "outsample",
                return_fitted = return_fitted,
            )
            for _d in test_data:
                state, _ = model.filter_update(state, _d)
                noise_ = _d - state[0]
                res.append(noise_)
            noise_ = pd.Series(res, index = test_data.index)

        elif self.method == "linear":
            fitted, model = self.linear_detrend(
                data = train_data,
                return_type = "outsample",
                return_fitted = return_fitted,
            )
            model: LinearRegression
            pred_x = np.array([np.ones(len(test_data)), 
                               np.arange(batch_size - test_size, batch_size)]).T
            pred = model.predict(pred_x)
            noise_ = test_data - pred
        
        elif self.method == "null":
            fitted = train_data
            noise_ = test_data

        return fitted, noise_
    
    def insample_detrend(self, data: pd.Series, method: str = "linear") -> pd.Series:
        self.method = method
        if method == "linear":
            return self.linear_detrend(
                data = data,
                return_type = "insample",
                return_fitted = True,
            )

        elif method == "isotonic":
            return self.isotonic_detrend(
                data = data,
                return_type = "insample",
                return_fitted = True,
            )
        
        elif method == "kf":
            return self.kalman_detrend(
                data = data,
                return_type = "insample",
                return_fitted = True,
            )
        
        else:
            raise ValueError("Invalid method. Must be one of 'linear', 'isotonic', 'kf'.")

    def plot_result(self,
                    index: pd.Series = None,
                    origin_data: pd.Series = None,
                    noise_data: pd.Series = None,
                    width: int = 2000, height: int = 800,
                    plot_fitted: bool = False) -> None:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        index = self.result.index if index is None else index
        origin_data = self.result[self.col] if origin_data is None else origin_data
        noise_data = self.result[self.noise_col] if noise_data is None else noise_data
        
        fig.add_trace(go.Scatter(x = index, 
                                 y = origin_data, 
                                 mode = 'lines', name = "origin",
                                 line=dict(color = 'blue')), row = 1, col = 1)

        fig.add_trace(go.Scatter(x = index, 
                                 y = noise_data, 
                                 mode = 'lines', name = "noise",
                                 line = dict(color = 'green')), row = 1, col = 1, 
                                 secondary_y=True)
        
        if plot_fitted:
            fig.add_trace(go.Scatter(x = index, 
                                y = origin_data - noise_data, 
                                mode = 'lines', name = "fitted",
                                line = dict(color = 'orange')), row = 1, col = 1, 
                                secondary_y=True)

        fig.update_layout(
            title="Origin Price vs Detrended Price",
            xaxis=dict(title="Time"),
            yaxis=dict(title=f"{self.col}", color="blue"),
            yaxis2=dict(title=f"{self.noise_col}", color="green", overlaying="y", side="right"),
            width=width,
            height=height
        )

        fig.update_xaxes(
            showspikes=True,
            spikecolor="grey",
            spikesnap="cursor",
            spikemode="across",
            spikedash="dash"
        )

        # 显示图表
        fig.show()

    """Detrend Part."""
    @staticmethod
    def detrend_series_ma(
            col: pd.Series, p: int = 12, 
            return_type: str = "detrend",
    ) -> pd.Series:
        raise NotImplementedError("Not implemented yet.")
        # col_use = col[col.first_valid_index() : col.last_valid_index()]
        
        # y_ = y.rolling(p, min_periods = 2).mean().\
        #     fillna(0).values

        # if return_type == "detrend":
        #     return pd.Series(y - y_, index = col_use.index, 
        #                     name = f"{col.name}_isotonic_detrend")
        # elif return_type == "predict":
        #     return pd.Series(y_, index = col_use.index, 
        #                 name = f"{col.name}_isotonic_predict")
        # else:
        #     raise ValueError("return_type must be detrend or predict")

    @staticmethod
    def linear_detrend(
            data: pd.Series,
            return_type: str = "outsample",
            return_fitted: bool = False,
    ) -> Union[pd.DataFrame, Union[pd.Series, LinearRegression]]:
        x = np.arange(len(data))
        x = np.array([np.ones(len(x)), x]).T
        reg = LinearRegression().fit(x, data.values)
        y_ = pd.Series(reg.predict(x), index = data.index) \
            if return_fitted else pd.Series()

        if return_type == "insample":
            """Insample detrend"""
            return pd.Series(data.values - y_, index = data.index)
        
        elif return_type == "outsample":
            return y_, reg
        
        else:
            raise ValueError("Invalid return_type. Must be one of 'insample' or 'outsample'.")
 
    @staticmethod
    def isotonic_detrend(
        data: pd.Series, 
        return_type: str = "outsample", 
        return_fitted: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.Series, IsotonicRegression]]:

        x = np.arange(len(data))
        ir = IsotonicRegression(out_of_bounds = "clip", increasing = "auto").\
            fit(x, data.values)
        
        y_ = ir.predict(x) if return_fitted else pd.Series()

        if return_type == "insample":
            """Insample detrend"""
            return pd.Series(data.values - y_, index = data.index)
        
        elif return_type == "outsample":
            return y_, ir
        
        else:
            raise ValueError("Invalid return_type. Must be one of 'insample' or 'outsample'.")
 
    @staticmethod
    def kalman_detrend(
        data: pd.Series, 
        return_type: str = "outsample", 
        return_fitted: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.Series, float, KalmanFilter]]:

        kf = KalmanFilter(initial_state_mean = 0, n_dim_obs = 1)
        filtered_state_means, _ = kf.filter(data.values)

        y_ = filtered_state_means.flatten() if return_fitted else pd.Series()
        if return_type == "insample":
            """Insample detrend"""
            return pd.Series(data.values - y_, index = data.index)
        
        elif return_type == "outsample":
            """Out sample detrend. Return a model, use this model to do the detrend."""
            return y_, filtered_state_means[-1], kf
        
        else:
            raise ValueError("Invalid return_type. Must be one of 'insample' or 'outsample'.")

#%%
if __name__ == "__main__":
    import polars as pl
    import pandas as pd
    from tqdm import tqdm
    import numpy as np
    import matplotlib.pyplot as plt
    """Generate random data"""
    n = 1000
    x = np.arange(n)
    trend = 0.1 * x + np.random.normal(scale=2, size=n)
    noise = np.random.normal(scale=3, size=n)
    data = trend + noise
    df = pd.DataFrame({"data": data})
    df["index"] = df.index
    plt.plot(df["data"])
    #%%
    """Insample detrending"""
    d = Detrendor(df, "data", "index", 
                    cumsum = False, window = 10, step = 2)
    noise = d.insample_detrend(df["data"], method = "kf")
    d.plot_result(index = df["index"], 
                origin_data = df["data"],
                noise_data = noise)
    #%%
    """Outsample (rolling predict) detrending"""
    d = Detrendor(df, "data", "index", 
                    cumsum = False, window = 10, step = 2)
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
            train_data = data[ : -d._step], 
            test_data = data[-d._step :], 
            method = "kf",
            return_fitted = False,
        )
        noise_list.append(noise)
    noise_result: pd.Series = pd.concat(noise_list)
    noise_result.name = d.noise_col
    d.noise = noise_result
    d.plot_result()

#%%
if __name__ == "__main__":
    """Linear First. Then KF."""
    import sys
    PROJ_DIR = "/home/ubuntu/CryptArb"
    sys.path.append(PROJ_DIR)
    import polars as pl
    from datetime import datetime
    import seaborn as sns
    from lib.data.DataLoader import DailyFileLoader
    import lib.helper.ut as ut
    sns.set_theme(style="darkgrid")

    df = pd.read_parquet(f"{PROJ_DIR}/data/bigdata/bn/5min/BTCUSDT_spot/BTCUSDT.parquet")
    df = df.loc[df["open_t_hk"] > ut.to_local_datetime("20231201")].reset_index(drop=True)

    pre_step = 20
    d = Detrendor(df.tail(400), "vwap", "close_t_hk", 
                    cumsum = False, window = 120, step = 1)
    fitted_list, noise_list = [], []
    group_data = d.create_rolling_datasets(
        d.data, 
        window = d._window, 
        step = d._step, 
        pre_step = pre_step,  # Use this for step1: linear detrend
    )
    for data in tqdm(group_data, desc = "Detrending...", 
                    total = (len(d.data) - d._window) // d._step):
        data: pd.Series

        linear_fitted_res = []
        batch_group_data = d.create_rolling_datasets(
            data, 
            window = pre_step + 1, 
            step = d._step, 
        )
        for batch_data in batch_group_data:
            fitted, noise = d.outsample_detrend(
                train_data = batch_data[ : -d._step], 
                test_data = batch_data[-d._step : ],  # Need to be the same with KF step.
                method = "linear",
            )
            linear_fitted_res.append(batch_data[-d._step : ] + noise)
        linear_fitted_res = pd.concat(linear_fitted_res)

        fitted, noise = d.outsample_detrend(
            train_data = linear_fitted_res[0 : -d._step], 
            test_data = linear_fitted_res[-d._step :], 
            method = "kf",
        )
        noise_list.append(noise)

    noise_result: pd.Series = pd.concat(noise_list)
    noise_result.name = d.noise_col
    d.noise = noise_result

    d.plot_result()
    
# if __name__ == "__main__":
#     """Kalman Filter Only."""
#     import sys
#     PROJ_DIR = "/home/ubuntu/CryptArb"
#     sys.path.append(PROJ_DIR)
#     import polars as pl
#     from datetime import datetime
#     import seaborn as sns
#     from lib.data.DataLoader import DailyFileLoader
#     import lib.helper.ut as ut
#     sns.set_theme(style="darkgrid")

#     df = pd.read_parquet(f"{PROJ_DIR}/data/bigdata/bn/5min/BTCUSDT_spot/BTCUSDT.parquet")
#     df = df.loc[df["open_t_hk"] > ut.to_local_datetime("20231201")].reset_index(drop=True)

#     d = Detrendor(df.tail(400), "vwap", "close_t_hk", 
#                   cumsum = False, window = 120, step = 5)
#     fitted_list, noise_list = [], []
#     group_data = d.create_rolling_datasets(
#         d.data, 
#         window = d._window, 
#         step = d._step, 
#     )
#     for data in tqdm(group_data, desc = "Detrending...", 
#                     total = (len(d.data) - d._window) // d._step):
#         data: pd.Series
#         fitted, noise = d.outsample_detrend(
#             train_data = data[pre_step : -d._step], 
#             test_data = data[-d._step :], 
#             method = "kf",
#             return_fitted = False,
#         )
#         noise_list.append(noise)

#     noise_result: pd.Series = pd.concat(noise_list)
#     noise_result.name = d.noise_col
#     d.noise = noise_result

#     d.plot_result()