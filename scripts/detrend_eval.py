#%%

import os
import sys

import pandas as pd
from pandarallel import pandarallel

pandarallel.initialize(progress_bar = True, nb_workers = 20)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.evaluate.Detrendor import (detrend_series_isotonic,
                                    detrend_series_linear, detrend_series_ma)
from lib.evaluate.Evaluator import ArbitrageEvaluator
from lib.helper.ut import get_monotonicity_degree, read_factor_return
from RA_Tasks.UnifiedArbitrage.lib.helper.plot_utils import (
    plot_detrend_compare, plot_detrend_perform)

__location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))


if __name__ == "__main__":
    df = read_factor_return(os.path.join(
        __location__, "../metadata/PredictorLSretWide.csv")
    )
    monotonicity_degree_df: pd.DataFrame = df.apply(
        lambda x: get_monotonicity_degree(x)
    ) # type: ignore
    col_use = monotonicity_degree_df.loc[
        (monotonicity_degree_df > 
        monotonicity_degree_df.quantile(0.9))
    ].index
    df = df[col_use].copy()
    ma_detrend_df: pd.DataFrame = df.apply(
        lambda x: detrend_series_ma(x, 12, "detrend")
    ) # type: ignore
    ma_predict_df: pd.DataFrame = df.apply(
        lambda x: detrend_series_ma(x, 12, "predict")
    ) # type: ignore
    linear_detrend_df: pd.DataFrame = df.apply(
        lambda x: detrend_series_linear(x, "detrend")
    ) # type: ignore
    linear_predict_df: pd.DataFrame = df.apply(
        lambda x: detrend_series_linear(x, "predict")
    ) # type: ignore
    isotonic_detrend_df: pd.DataFrame = df.apply(
        lambda x: detrend_series_isotonic(x, "detrend")
    ) # type: ignore
    isotonic_predict_df: pd.DataFrame = df.apply(
        lambda x: detrend_series_isotonic(x, "predict")
    ) # type: ignore
    """Plotting Cell"""
    fig = plot_detrend_perform(
        df.columns[0], ma_detrend_df, linear_detrend_df, isotonic_detrend_df,
        ma_predict_df, linear_predict_df, isotonic_predict_df,
        df
    )
    e = ArbitrageEvaluator()
    eval_matrix: pd.DataFrame = isotonic_detrend_df.parallel_apply(
        lambda x: e.eval_arbitrage_award(x, kurt_hate=3, mrs_factor=1)
    ).T # type: ignore
    eval_matrix = eval_matrix.reset_index()
    eval_matrix.columns = ["factor", "utility", "mrs_value", "std_value", "nsr_value", "excess_kurt", "lag_use", "root"]
    eval_matrix.to_csv("/home/sida/YIPENG/Task2_FactorTiming/results/Isotonic_denoise_analysis.csv", index = None) # type: ignore
    eval_matrix = eval_matrix.sort_values(by = "utility", ascending = False)

    fig = plot_detrend_compare(
        list(eval_matrix["factor"][:3]), ma_detrend_df, isotonic_detrend_df,
        df, isotonic_predict_df
    )
# %%
