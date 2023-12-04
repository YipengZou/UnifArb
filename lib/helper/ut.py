import pandas as pd
import numpy as np

def read_factor_return(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.index = pd.to_datetime(df["date"]) # type: ignore
    df.drop("date", axis=1, inplace=True)
    return df

def get_monotonicity_degree(col: pd.Series) -> float:
    col_use = col[col.first_valid_index() : col.last_valid_index()]
    vals, unique = np.unique(np.sign(col_use), return_counts = True)
    pos_prop = pd.Series(unique, vals)[1] / len(col_use)
    return max(pos_prop, 1 - pos_prop)