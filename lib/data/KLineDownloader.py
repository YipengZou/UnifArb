import polars as pl
from datetime import datetime
import okx.MarketData as MarketData
import os
import time
from loguru import logger
import abc
import time
import math
from datetime import datetime
from typing import Union
import pandas as pd

class DataDownloader(abc.ABC):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def get(self):
        pass

    @staticmethod
    def cur_ts():
        """Current timestamp"""
        cur_t = time.localtime()
        cur_t = time.struct_time((
            cur_t.tm_year, cur_t.tm_mon, cur_t.tm_mday, 
            cur_t.tm_hour, cur_t.tm_min, 0, 
            cur_t.tm_wday, cur_t.tm_yday, cur_t.tm_isdst
        ))
        return DataDownloader.time2stamp(cur_t)
    
    @staticmethod
    def time2stamp(t: Union[str, time.struct_time]) -> int:
        if isinstance(t, time.struct_time):
            return int(time.mktime(t))
        
        elif isinstance(t, str):
            t_format = "%Y-%m-%d %H:%M:%S"
            t = pd.to_datetime(t).strftime(t_format)
            return int(time.mktime(time.strptime(t, t_format)))
        
    @staticmethod
    def stamp2time(stamp: int) -> str:
        if math.floor(math.log10(stamp)) == 12:  # Handle milliseconds
            stamp /= 1000

        return datetime.fromtimestamp(stamp).strftime("%Y-%m-%d %H:%M:%S")
    
class OKXKLineDownloader(DataDownloader):
    def __init__(self, earliest_time: str = "20230101",
                 market_flag: str = "0") -> None:
        self.earliest_time = self.time2stamp(earliest_time)
        self.marketDataAPI =  MarketData.MarketAPI(
            flag = market_flag
        )  # Real market:0 , simulation market：1
    
    def get(self, end_ts: int = DataDownloader.cur_ts(), 
            inst: str = "BTC-USDT", freq: str = "1m",
            past_n: int = 100) -> pl.DataFrame:

            result = self.marketDataAPI.get_history_candlesticks(
                instId = inst, 
                after = end_ts,  # Last timestamp
                bar = freq,
                limit = past_n,
            )
            df = pl.DataFrame(result["data"]).transpose()
            df.columns = [
                "ts", "open", "high", "low", "close", 
                "volume", "volccy", "volccyquote", "confirm"
            ]
            df = df.cast(pl.Float64)
            df = df.with_columns(
                trade_time = (pl.col("ts").cast(pl.Int64) / 1000)
                    .map_elements(lambda x: datetime.fromtimestamp(x).strftime("%H:%M:%S")),
                TradeDate = (pl.col("ts").cast(pl.Int64) / 1000)
                    .map_elements(lambda x: int(datetime.fromtimestamp(x).strftime("%Y%m%d"))),
                inst = pl.lit(inst),
            )
            return df
    
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    logger.add("../../logger/KLineDownloader.log", rotation="1 day", retention="7 days")
    save_path = "../../data/bigdata/5min/BTC_USDT"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    record = pl.DataFrame()
    d = OKXKLineDownloader()
    end_ts = DataDownloader.cur_ts() * 1000

    while True:
        try:
            df = d.get(end_ts = int(end_ts),
                    freq = "5m")
            record = record.vstack(df)
            if min(df["TradeDate"]) < 20200101:
                break

            if min(record["TradeDate"]) != max(record["TradeDate"]):
                trade_date = max(record["TradeDate"])
                record.filter(pl.col("TradeDate") == trade_date)\
                    .write_parquet(f"{save_path}/{trade_date}.parquet")
                record = record.filter(pl.col("TradeDate") < trade_date)

            end_ts = min(record["ts"])
            time.sleep(2/10)
            logger.success((min(record["TradeDate"]), max(record["TradeDate"]), end_ts))

        except:
            time.sleep(2/20)
            logger.exception("Error")
            continue