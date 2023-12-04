#%%
import pandas as pd
import numpy as np
import os
import sys
from typing import List
sys.path.append("/home/sida/YIPENG/RA_Tasks/UnifiedArbitrage")
from lib.evaluate.Detrendor import detrend_series_isotonic
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import warnings
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)
        

def plot_res(pg: PolicyGenerator, last_n: int = None): # type: ignore
    df = pg.move_df.copy()
    if last_n is not None:
        df = df.iloc[-last_n:]

    df.index = df.index.map(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    pnl_list = pd.Series(pg.res.loc[0, "pnl_list"], 
                         index = pg.res.loc[0, "s_date"]) # type: ignore
    pnl_list.index = pnl_list.index.map(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    df["pnl"] = pnl_list
    df["pnl"] = df["pnl"].fillna(0)
    df["pnl"] = df["pnl"].cumsum()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(24, 12), sharex=True)

    ax1.plot(df[pg._col], label='Return', color='blue')
    ax1.plot(df[f"boll_u"], label='boll_u', color='grey', linestyle = "--", alpha = 0.7)
    ax1.plot(df[f"boll_l"], label='boll_l', color='grey', linestyle = "--", alpha = 0.7)

    ax2.plot(df[pg.dcol], label='Noise', color='orange', lw = 2)
    ax2.plot(df[f"u_bound"], label='u_bound', color='grey', linestyle = "--", alpha = 0.7)
    ax2.plot(df[f"l_bound"], label='l_bound', color='grey', linestyle = "--", alpha = 0.7)
    
    ax3.plot(df["pnl"], label='PnL', color='green')
    ax3.plot(df["pnl"] - df[pg._col], label='excess return', color='pink')
    
    """Anotate buy and sell points"""
    for date in pg.res.loc[0, "b_date"]: # type: ignore
        date = datetime.strptime(date, "%Y-%m-%d")
        if date < df.index[0]:
            continue
        ax1.scatter(date, df[pg._col][date], c='red', marker='o')
        ax2.scatter(date, df[pg.dcol][date], c='red', marker='o')

    for date in pg.res.loc[0, "s_date"]: # type: ignore
        date = datetime.strptime(date, "%Y-%m-%d")
        if date < df.index[0]:
            continue
        ax1.scatter(date, df[pg._col][date], c='green', marker='o')
        ax2.scatter(date, df[pg.dcol][date], c='green', marker='o')
    
    for date in pg.res.loc[0, "stop_loss_date"]: # type: ignore
        date = datetime.strptime(date, "%Y-%m-%d")
        if date < df.index[0]:
            continue
        ax1.scatter(date, df[pg._col][date], c='blue', marker='o')
        ax2.scatter(date, df[pg.dcol][date], c='blue', marker='o')

    # 设置子图标题和标签
    for ax in [ax1, ax2, ax3]:
        ax.legend()
        ax.grid()

    ax1.set_title('Return Over Time')
    ax2.set_title('Detrend Return Over Time')
    ax3.set_title(f"PnL Over Time:\
                  a: {pg.res.loc[0, 'a']}, b: {pg.res.loc[0, 'b']}, return: {pg.res.loc[0, 'return']}%\n\
                    Winning rate: {pg.res.loc[0, 'winning_rate']}%, avg hold period: {pg.res.loc[0, 'avg_hold_p']}days, avg return: {pg.res.loc[0, 'avg_return']}%")

#%%
col = "volatility_onemonthv"
dt = Detrendor(col, step = 6,
               data_path = "/home/sida/YIPENG/RA_Tasks/UnifiedArbitrage/metadata/CN_daily_return.csv",
               start_date = 20150101,
               end_date = 20230101,)
fig = dt.plot_detrend()
#%%
pg = PolicyGenerator(dt)
pg.grid_search(np.linspace(0, 2, 10), np.linspace(1, 3, 20))
plot_res(pg)
# plot_res(pg, last_n = 100)
# %%

# policy -> based on volatily -> outsample  # hit but not cleared -> hold until clear
# 止损点: ATR (if open -> stop loss criteria. control the drawdown. Based on historical volatility.)
## ART upper replace 0.

# utilit -> 过去10年utility. 
# monthly -> daily
# a, b constant + volatility

# Yuxiao CHian 80 factors. most volatile. 
# 白银+白银ETF数据 / 黄金矿业ETF + 黄金

# %%
