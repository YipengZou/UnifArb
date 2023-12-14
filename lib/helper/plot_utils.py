import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List
from datetime import datetime

from ..evaluate.PolicyGenerator import PolicyGenerator

def plot_pg_res(pg: PolicyGenerator, last_n: int = -1): 
    df = pg.best_bt.move_df.copy()
    if last_n > 0:
        df = df.iloc[-last_n:]

    if isinstance(df.index[0], str):
        df.index = df.index.map(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    
    pnl_list = pd.Series(pg.best_result["pnl_list"], 
                         index = pg.best_result["s_date"]) 
    pnl_list.index = pnl_list.index.map(lambda x: pd.to_datetime(x))
    df["pnl"] = pnl_list
    df["pnl"] = df["pnl"].fillna(0)
    df["pnl"] = df["pnl"].cumsum()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(24, 12), sharex=True)

    ax1.plot(df[pg._col], label='Return', color='blue')
    ax1.plot(df[f"stop_earning_bar"], label='stop_earning_bar', color='grey', linestyle = "--", alpha = 0.7)
    ax1.plot(df[f"stop_loss_bar"], label='stop_loss_bar', color='grey', linestyle = "--", alpha = 0.7)

    ax2.plot(df[pg.dcol], label='Noise', color='orange', lw = 2)
    ax2.plot(df[f"u_bound"], label='u_bound', color='grey', linestyle = "--", alpha = 0.7)
    ax2.plot(df[f"l_bound"], label='l_bound', color='grey', linestyle = "--", alpha = 0.7)
    
    ax3.plot(df["pnl"], label='PnL', color='green')
    ax3.plot(df["pnl"] - df[pg._col], label='excess return', color='pink')
    
    """Anotate buy and sell points"""
    for date in pg.best_result["b_date"]: 
        date = pd.to_datetime(date)
        if date < df.index[0]:
            continue
        ax1.scatter(date, df[pg._col][date], c='red', marker='o')
        ax2.scatter(date, df[pg.dcol][date], c='red', marker='o')

    for date in pg.best_result["s_date"]: 
        date = pd.to_datetime(date)
        if date < df.index[0]:
            continue
        ax1.scatter(date, df[pg._col][date], c='green', marker='o')
        ax2.scatter(date, df[pg.dcol][date], c='green', marker='o')
    
    for date in pg.best_result["stop_loss_date"]: 
        date = pd.to_datetime(date)
        if date < df.index[0]:
            continue
        ax1.scatter(date, df[pg._col][date], c='blue', marker='o')
        ax2.scatter(date, df[pg.dcol][date], c='blue', marker='o')
    
    for date in pg.best_result["stop_earning_date"]: 
        date = pd.to_datetime(date)
        if date < df.index[0]:
            continue
        ax1.scatter(date, df[pg._col][date], c='orange', marker='o')
        ax2.scatter(date, df[pg.dcol][date], c='orange', marker='o')


    # 设置子图标题和标签
    for ax in [ax1, ax2, ax3]:
        ax.legend()
        ax.grid()

    ax1.set_title('Return Over Time')
    ax2.set_title('Detrend Return Over Time')
    ax3.set_title(f"PnL Over Time:\
                  a: {pg.best_result['a']}, b: {pg.best_result['b']}, return: {pg.best_result['return']}%\n\
                    Winning rate: {pg.best_result['winning_rate']}%, avg hold period: {pg.best_result['avg_hold_p']}days, avg return: {pg.best_result['avg_return']*100}%\n\
                        Annualized Return: {pg.best_result['ann_return']*100}%")
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