import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ..evaluate.PolicyGenerator import PolicyGenerator
def plot_pg_res(pg: PolicyGenerator):
    df = pg.best_bt.move_df.copy()
    pnl_list = pd.Series(pg.best_result["pnl_list"], 
                            index = pg.best_result["s_date"],
                            name = "pnl") 
    if isinstance(pnl_list.index[0], str):
        pnl_list.index = pnl_list.index.map(lambda x: pd.to_datetime(x))
    df["pnl"] = pnl_list
    df["pnl"] = df["pnl"].fillna(0)
    df["pnl"] = df["pnl"].cumsum()

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
    b_date = [pd.to_datetime(x) for x in pg.best_result["b_date"]]
    s_date = [pd.to_datetime(x) for x in pg.best_result["s_date"]]
    sl_date = [pd.to_datetime(x) for x in pg.best_result["stop_loss_date"]]
    se_date = [pd.to_datetime(x) for x in pg.best_result["stop_earning_date"]]

    """Figure 1"""
    fig.add_trace(go.Scatter(x = df.index, y = df[pg._col].values, 
                            mode='lines', name = pg._col,
                            line = dict(color='blue')
                            ), row=1, col=1)
    fig.add_trace(go.Scatter(x = df.index, y = df[f"stop_earning_bar"].values, 
                            mode='lines', name='stop_earning_bar',
                            line = dict(color='grey', dash='dash')
                            ), row=1, col=1)
    fig.add_trace(go.Scatter(x = df.index, y = df[f"stop_loss_bar"].values, 
                            mode='lines', name='stop_loss_bar',
                            line = dict(color='grey', dash='dash')
                            ), row=1, col=1)
    fig.add_trace(go.Scatter(x = b_date, y = df.loc[b_date, pg._col].values,
                            mode='markers', name='buy',
                            marker = dict(size = 10, color = "red")
                            ), row=1, col=1)  # buy
    fig.add_trace(go.Scatter(x = s_date, y = df.loc[s_date, pg._col].values,
                            mode='markers', name='sell',
                            marker = dict(size = 10, color = "green")
                            ), row=1, col=1)  # sell
    fig.add_trace(go.Scatter(x = se_date, y = df.loc[se_date, pg._col].values,
                            mode='markers', name='stop earning',
                            marker = dict(size = 10, color = "orange")
                            ), row=1, col=1)  # stop earning
    fig.add_trace(go.Scatter(x = sl_date, y = df.loc[sl_date, pg._col].values,
                            mode='markers', name='stop loss',
                            marker = dict(size = 10, color = "blue")
                            ), row=1, col=1)  # stop loss

    """Figure 2"""
    fig.add_trace(go.Scatter(x = df.index, y = df[pg.dcol].values, 
                            mode='lines', name = pg.dcol,
                            line = dict(color='orange')
                            ), row=2, col=1)
    fig.add_trace(go.Scatter(x = df.index, y = df[f"u_bound"].values, 
                            mode='lines', name='u_bound',
                            line = dict(color='grey', dash='dash')
                            ), row=2, col=1)
    fig.add_trace(go.Scatter(x = df.index, y = df[f"l_bound"].values, 
                            mode='lines', name='l_bound',
                            line = dict(color='grey', dash='dash')
                            ), row=2, col=1)
    fig.add_trace(go.Scatter(x = b_date, y = df.loc[b_date, pg.dcol].values,
                            mode='markers', name='buy',
                            marker = dict(size = 10, color = "red")
                            ), row=2, col=1)  # buy
    fig.add_trace(go.Scatter(x = s_date, y = df.loc[s_date, pg.dcol].values,
                            mode='markers', name='sell',
                            marker = dict(size = 10, color = "green")
                            ), row=2, col=1)  # sell
    fig.add_trace(go.Scatter(x = se_date, y = df.loc[se_date, pg.dcol].values,
                            mode='markers', name='stop earning',
                            marker = dict(size = 10, color = "orange")
                            ), row=2, col=1)  # stop earning
    fig.add_trace(go.Scatter(x = sl_date, y = df.loc[sl_date, pg.dcol].values,
                            mode='markers', name='stop loss',
                            marker = dict(size = 10, color = "blue")
                            ), row=2, col=1)  # stop loss

    """Figure 3"""
    fig.add_trace(go.Scatter(x = df.index, y = df["pnl"], 
                            mode='lines', name = "PnL",
                            line = dict(color='orange')
                            ), row=3, col=1)

    fig.update_layout(
                    xaxis_title='x',
                    yaxis_title='y',
                    hovermode='x',
                    legend = dict(x=1, y=1),
                    width = 2000, height = 1500)
    fig.show()
    return fig

def plot_detrend_perform(col: str, 
                         ma_dt_df, ln_dt_df, iso_dt_df,
                         ma_pr_df, ln_pr_df, iso_pr_df,
                         df) -> Figure:
    fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(12, 12))
    plt.figure(figsize = (12, 8))
    ax0.plot(ma_dt_df[col], label = "MA Detrend")
    ax0.plot(ln_dt_df[col], label = "Linear Detrend")
    ax0.plot(iso_dt_df[col], label = "Isotonic Detrend")
    ax0.legend()
    ax0.set_title(f"Detrend Cummulative Return for Factor: {col}")

    ax1.plot(ma_pr_df[col], label = "MA Predict")
    ax1.plot(ln_pr_df[col], label = "Linear Predict")
    ax1.plot(iso_pr_df[col], label = "Isotonic Predict")
    ax1.plot(df[col].cumsum(), label = "Original")
    ax1.legend()
    ax1.set_title(f"Predict Cummulative Return for Factor: {col}")
    plt.show()

    return fig

def plot_detrend_compare(cols: List[str],
                         ma_dt_df, iso_dt_df,
                         df, iso_pr_df):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(12, 18))
    for col in cols:
        ax0.plot(iso_dt_df[col], label = f"{col} Isotonic Detrend")
        ax0.plot(ma_dt_df[col], label = f"{col} MA Detrend")
        ax1.plot(df[col].cumsum(), label = f"{col} Original Cummulative Return")
        ax2.plot(iso_pr_df[col], label = f"{col} Isotonic Predict")

    ax0.set_title(f"Detrend Cummulative Return")
    ax1.set_title(f"Originial Cummulative Return")
    ax2.set_title(f"Predict Cummulative Return")
    ax0.legend()
    ax1.legend()
    ax2.legend()
    plt.show()

    return fig