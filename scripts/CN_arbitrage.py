#%%
import os
import sys
from typing import List

import numpy as np
import pandas as pd
from scipy.datasets import face

sys.path.append("/home/zouyipeng/Workspace/UnifiedArb")
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
from lib.evaluate.Detrendor import Detrendor, detrend_series_isotonic
from lib.evaluate.Evaluator import ArbitrageEvaluator
from lib.evaluate.PolicyGenerator import PolicyGenerator
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

def plot_res(pg: PolicyGenerator, last_n: int = None): # type: ignore
    df = pg.best_bt.move_df.copy()
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
    return fig

def save_plots(cols, det_plots, res_plots, res_sample_plots, eval_plots):
    is_init = False
    w1, h1 = Image.open(det_plots[0]).size
    c = canvas.Canvas(pdf_path, pagesize = (0, 0))
    for _col, p1, p2, p3, p4 in zip(cols, res_sample_plots, res_plots, det_plots, eval_plots):
        w1, h1 = Image.open(p1).size
        w2, h2 = Image.open(p2).size
        w3, h3 = Image.open(p3).size
        w4, h4 = Image.open(p4).size

        page_w = max(w1, w2, w3, w4)
        page_h = h1 + h2 + h3 + h4 + 50
        if not is_init:
            c = canvas.Canvas(pdf_path, pagesize = (page_w, page_h))
            is_init = True

        c.drawImage(p1, 0, 0, width = w1, height = h1)
        c.drawImage(p2, 0, h1, width = w2, height = h2)
        c.drawImage(p3, 0, h1+h2, width = w3, height = h3)
        c.drawImage(p4, (page_w - w4)/2, h1+h2+h3, width = w4, height = h4)
        
        c.setFont("Helvetica", 40)
        c.drawString(0, page_h - 40, f"Factor: {_col}")
        c.showPage()
    c.save()

#%%
if __name__ == "__main__":
    factor_path = "/home/zouyipeng/Workspace/UnifiedArb/metadata/CN_daily_return.csv"
    save_folder = "/home/zouyipeng/Workspace/UnifiedArb/results/plots"
    pdf_path = "/home/zouyipeng/Workspace/UnifiedArb/results/CN_plot.pdf"
    cols = pd.read_csv(factor_path).columns[1:]
    det_plots, res_plots, res_sample_plots, eval_plots = [], [], [], []
    e = ArbitrageEvaluator()
    for col in tqdm(cols, desc = "Evaluating CN..."):
        dt = Detrendor(col, step = 1,
                    data_path = factor_path,
                    start_date = 20200101,
                    end_date = 20230101,)
        fig = dt.plot_detrend()
        fig_path = os.path.join(save_folder, f"{col}_detrend.png")
        fig.savefig(fig_path, bbox_inches='tight', facecolor = "white")
        det_plots.append(fig_path)
        
        pg = PolicyGenerator(dt)
        pg.grid_search(np.linspace(0, 3, 20), np.linspace(1, 5, 20))

        res_plot = plot_res(pg)
        res_path = os.path.join(save_folder, f"{col}_res.png")
        res_plot.savefig(res_path, bbox_inches='tight', facecolor = "white")
        res_plots.append(res_path)

        res_sample = plot_res(pg, last_n = 200)
        res_sample_path = os.path.join(save_folder, f"{col}_res_sample.png")
        res_sample.savefig(res_sample_path, bbox_inches='tight', facecolor = "white")
        res_sample_plots.append(res_sample_path)

        eval_res = pd.DataFrame(e.eval_arbitrage_award(dt.preds)).T.round(4)
        eval_res.columns = ['utility', 'mrs_value', 'std_value', 'nsr_value', 'excess_kurt', 'lag_use', 'root']
        plt.figure(figsize=(12, 4))
        ax = plt.gca()
        ax.axis('off')
        table = plt.table(cellText=eval_res.values, 
                        colLabels=eval_res.columns.tolist(), loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(20)
        table.scale(2, 2)  # 调整表格大小
        eval_res_path = os.path.join(save_folder, f"{col}_eval_res.png")
    #     plt.savefig(eval_res_path, bbox_inches='tight', facecolor = "white")
    #     eval_plots.append(eval_res_path)
        
    # save_plots(cols, det_plots, res_plots, res_sample_plots, eval_plots)

# %%
