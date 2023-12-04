import pandas as pd

class DataLoader:
    def __init__(self, path: str = "/home/sida/YIPENG/Task2_FactorTiming/data/PredictorLSretWide.csv",
                 ) -> None:
        self._df = pd.read_csv(path).sort_values(by = "date")
        self._calendar = list(self._df["date"])

    def get(self, col: str, date: str, backward: int, skip_last: int = 1) -> pd.Series:
        """
            Skip_last: avoid forward looking. Ignore last data.
        """
        end_idx = self._calendar.index(date)
        start_idx = max(end_idx - backward, 0)
        rets = self._df[col][start_idx : end_idx - skip_last]
        days = self._calendar[start_idx : end_idx - skip_last]
        return pd.Series(rets.values, index = days, name = col)
    
    def get_full_col(self, col: str) -> pd.Series:
        res = self._df[col]
        res.index = self._df["date"].values # type: ignore
        return res
    
    @property
    def calendar(self):
        return self._calendar