import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List



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