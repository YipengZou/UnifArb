import polars as pl
import os
import abc

class DataLoader(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def get(self):
        pass

class DailyFileLoader(DataLoader):
    def __init__(self, dir: str):
        self._dir = dir
    
    def __repr__(self) -> str:
        return self.info.__repr__()

    @property
    def info(self) -> pl.DataFrame:
        if not hasattr(self, "_info"):
            files = os.listdir(self._dir)
            date = [int(file.split(".")[0]) for file in files]
            paths = [f"{self._dir}/{f}" for f in files]
            size = [os.path.getsize(f)/1024**2 for f in paths]
            self._info = pl.DataFrame(
                {
                    "date": date,
                    "file": paths,
                    "size_mb": size
                }
            ).sort("date")
        
        return self._info

    def get(self, start: int, end: int):
        paths = self.info.filter(pl.col("date").is_between(start, end))["file"]
        return pl.read_parquet(paths).sort(by = ["TradeDate", "trade_time"])