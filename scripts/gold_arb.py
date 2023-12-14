# %%
import os
import sys
from typing import List
import matplotlib.pyplot as plt
PROJ_DIR = os.path.abspath(__file__).split("/scripts/")[0]
sys.path.append(PROJ_DIR)
from lib.evaluate.Detrendor import Detrendor
from lib.evaluate.PolicyGenerator import PolicyGenerator
from lib.helper.plot_utils import plot_pg_res
from datetime import datetime
import polars as pl
import pandas as pd
from loguru import logger
df = pl.read_csv("/home/sida/YIPENG/Task2_FactorTiming/data/gold_futures_comex/td_comex_xau_cnhusd_210104.csv")
df = df.with_columns(
    td_close_USD = pl.col("td_close") * 31.1035 * pl.col("CNHUSD_close")
)
df = df.with_columns(
    com_td_diff = pl.col("td_close_USD") - pl.col("comex_close"),
    xau_td_diff = pl.col("td_close_USD") - pl.col("xau_close"),
    xau_com_diff = pl.col("td_close_USD") - pl.col("xau_close")
)
df = df.head(500000)
df = df.to_pandas()
df.index = df["date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
df = df.resample("30T").first()
logger.success("Finish data processing.")
# %%

#%%
dt = Detrendor(
    col = "com_td_diff",
    window = 300,
    step = 20,
    d_method = "kf",
    start_date = 20210105,
    end_date = 20211005,
    data = df["com_td_diff"], # type: ignore
)
dt.batch_detrend()

# %%
pg = PolicyGenerator(dt)
pg.bayes_search(n_trials = 30)
pg.best_result.keys()
pg.best_result["pnl_list"]
fig = plot_pg_res(pg)
# %%
import polars as pl
#%%