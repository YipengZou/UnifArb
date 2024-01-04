from ..simulator.BackTestor import BackTestor
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

class BackTestorPlotter:
    def __init__(self, bt: BackTestor) -> None:
        self._bt = bt
        
    @property
    def bt(self):
        return self._bt
    
    @property
    def pnl(self) -> pd.Series:
        return self.bt.pnl
    
    @property
    def data(self) -> pd.DataFrame:
        if not hasattr(self, "_data"):
            self._data = self.bt.data.copy()
            self._data["pnl"] = self.pnl
        return self._data
    
    def plot(self):
        fig = make_subplots(rows = 3, cols = 1, shared_xaxes = True)
        self.plot_price_line(fig = fig, row = 1, col = 1)
        self.plot_bs_point(fig = fig, row = 1, col = 1)
        self.plot_price_stop_loss(fig = fig, row = 1, col = 1)
        self.plot_price_stop_earning(fig = fig, row = 1, col = 1)
        self.plot_signal(fig = fig, row = 2, col = 1)
        self.plot_signal_bound(fig = fig, row = 2, col = 1)
        self.plot_signal_bs_point(fig = fig, row = 2, col = 1)
        self.plot_pnl(fig = fig, row = 3, col = 1)
        self.update_layout(fig = fig)

        return fig


    def plot_price_line(self, fig: go.Figure, row: int = 1, col: int = 1):
        low_vol_list = self.bt.low_vol_list.reindex(self.bt.valid_index)
        fig.add_trace(
                go.Scatter(
                    x = self.bt.valid_index, 
                    y = self.data.loc[self.bt.valid_index, self.bt.price_col],
                    mode = 'lines', name = f"{self.bt.price_col}_hvol",
                    line = dict(color = 'blue')
                ), row = row, col = col,
            )
        fig.add_trace(
                go.Scatter(
                    x = low_vol_list.index, 
                    y = low_vol_list.values,
                    mode = 'lines', name = f"{self.bt.price_col}_lvol",
                    line = dict(color = 'grey')
                ), row = row, col = col,
            )
    
    def plot_bs_point(self, fig: go.Figure, row: int = 1, col: int = 1):

        fig.add_trace(go.Scatter(
            x = self.bt.summary["buy_time"], 
            y = self.data.loc[self.bt.summary["buy_time"], 
                              self.bt.price_col].values,
            mode = 'markers', name = 'buy',
            marker = dict(size = 10, color = "red")
        ), row = row, col = col)  # buy

        fig.add_trace(go.Scatter(
            x = self.bt.summary["sell_time"], 
            y = self.data.loc[self.bt.summary["sell_time"],
                              self.bt.price_col].values,
            mode = 'markers', name = 'sell',
            marker = dict(size = 10, color = "green")
        ), row = 1, col = 1)  # sell

    def plot_price_stop_loss(self, fig: go.Figure, row: int = 1, col: int = 1):
        sig_bound: pd.DataFrame = np.exp(self.bt.signal_summary.copy())
        sig_bound = pd.merge(sig_bound, self.data[self.bt.price_col], 
                            left_index = True, right_index = True, how = "left")
        sig_bound = sig_bound.eval(f"price_low = low_bound * {self.bt.price_col}")
        fig.add_trace(
            go.Scatter(
                x = sig_bound.index, y = sig_bound["price_low"].shift(1),
                mode = 'lines', name = 'l_bound',
                line = dict(color = 'grey', dash = 'dash')
            ), row = row, col = col
        )
        fig.add_trace(
            go.Scatter(
                x = self.bt.stop_loss_summary["stop_loss_time"], 
                y = self.bt.stop_loss_summary["stop_loss_price_list"].values,
                mode = 'markers', name = 'stop loss',
                marker = dict(size = 10, color = "orange")
            ), row = row, col = col
        )  # stop earning

    def plot_price_stop_earning(self, fig: go.Figure, row: int = 1, col: int = 1):
        return
        # //TODO: Add stop earning
        fig.add_trace(
            go.Scatter(
                x = sig_bound.index, y = sig_bound["price_up"].shift(1),
                mode = 'lines', name = 'u_bound',
                line = dict(color = 'grey', dash = 'dash')
            ), row = row, col = col,
        )
        fig.add_trace(go.Scatter(x = sl_date, y = df_use.loc[sl_date, pg._col].values,
                                mode='markers', name='stop loss',
                                marker = dict(size = 10, color = "blue")
                                ), row=1, col=1)  # stop loss
    
    def plot_signal(self, fig: go.Figure, row: int = 2, col: int = 1):
        fig.add_trace(
            go.Scatter(
                x = self.bt.valid_index, 
                y = self.data.loc[self.bt.valid_index, self.bt.signal_col], 
                mode = 'lines', name = self.bt.signal_col,
                line = dict(color = 'orange')
            ), row = row, col = col,
        )

        # fig.add_trace(
        #         go.Scatter(
        #             x = self.bt.high_vol_list.index, 
        #             y = self.bt.high_vol_list.values, 
        #             mode = 'lines', name = f"{self.bt.price_col}_lvol",
        #             line = dict(color = 'grey')
        #         ), row = row, col = col,
        #     )

    def plot_signal_bound(self, fig: go.Figure, row: int = 2, col: int = 1):
        fig.add_trace(
            go.Scatter(
                x = self.bt.bs_summary.index, y = self.bt.bs_summary["up_bound"],
                mode = 'lines', name = 'u_bound',
                line = dict(color = 'grey', dash = 'dash')
            ), row = row, col = col,
        )
        fig.add_trace(
            go.Scatter(
                x = self.bt.bs_summary.index, y = self.bt.bs_summary["low_bound"],
                mode = 'lines', name = 'l_bound',
                line = dict(color = 'grey', dash = 'dash')
            ), row = row, col = col,
        )

    def plot_signal_bs_point(self, fig: go.Figure, row: int = 2, col: int = 1):
        fig.add_trace(go.Scatter(
            x = self.bt.summary["buy_time"], 
            y = self.data.loc[self.bt.summary["buy_time"], self.bt.signal_col].values,
            mode = 'markers', name = 'buy',
            marker = dict(size = 10, color = "red")
        ), row = row, col = col)  # buy
        fig.add_trace(go.Scatter(
            x = self.bt.summary["sell_time"], 
            y = self.data.loc[self.bt.summary["sell_time"], self.bt.signal_col].values,
            mode = 'markers', name = 'sell',
            marker = dict(size = 10, color = "green")
        ), row = row, col = col)  # sell

    def plot_signal_stop_point(self, fig: go.Figure, row: int = 2, col: int = 1):
        return 
        # // TODO: Add stop point
        # fig.add_trace(go.Scatter(x = s_date, y = df_use.loc[s_date, pg.dcol].values,
        #                         mode='markers', name='sell',
        #                         marker = dict(size = 10, color = "green")
        #                         ), row=2, col=1)  # sell
        # fig.add_trace(go.Scatter(x = se_date, y = df_use.loc[se_date, pg.dcol].values,
        #                         mode='markers', name='stop earning',
        #                         marker = dict(size = 10, color = "orange")
        #                         ), row=2, col=1)  # stop earning
        # fig.add_trace(go.Scatter(x = sl_date, y = df_use.loc[sl_date, pg.dcol].values,
        #                         mode='markers', name='stop loss',
        #                         marker = dict(size = 10, color = "blue")
        #                         ), row=2, col=1)  # stop loss
    
    def plot_pnl(self, fig: go.Figure, row: int = 3, col: int = 1):
        fig.add_trace(
            go.Scatter(
                x = self.bt.valid_index, 
                y = self.data.loc[self.bt.valid_index, "pnl"], 
                mode = 'lines', name = "PnL",
                line = dict(color = 'orange')
            ), row = row, col = col,
        )

    def update_layout(self, fig: go.Figure):
        fig.update_layout(
            xaxis_title = 'x',
            yaxis_title = 'y',
            hovermode = 'x',
            legend = dict(x = 1, y = 1),
            width = 2000, height = 1500
        )

        fig.update_xaxes(
            showspikes = True,
            spikecolor = "grey",
            spikesnap = "cursor",
            spikemode = "across",
            spikedash = "dash"
        )