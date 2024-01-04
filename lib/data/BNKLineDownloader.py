#%%
import sys
import os
import os.path as p
import time
from datetime import datetime
from typing import List



import pandas as pd
from loguru import logger
from pandarallel import pandarallel
from binance.enums import HistoricalKlinesType
from binance.client import Client
pandarallel.initialize(progress_bar = False)

PROJ_DIR = p.dirname(p.dirname(p.dirname(p.abspath(__file__))))
sys.path.append(PROJ_DIR)
from api.bn import API, SEC_KEY

class BNKLineDownloader:
    tz = "Asia/Hong_Kong"
    _client = Client(API, SEC_KEY)

    def __init__(self) -> None:
        pass

    @property
    def client(self) -> Client:
        return self._client
    
    @property
    def server_time(self) -> pd.Timestamp:
        return pd.to_datetime(
            self.client.get_server_time()["serverTime"], 
            utc=True, unit = "ms",
        ).tz_convert(self.tz)
    
    @staticmethod
    def kline_converter(klines: List[List], tz: str = "Asia/Hong_Kong") -> pd.DataFrame:
        suffix = "hk" if tz == "Asia/Hong_Kong" else tz.split("/")[1]
        df = pd.DataFrame(klines)
        df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                    'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 
                    'taker_buy_quote_asset_volume', 'ignore']
        
        df[f"open_t_{suffix}"] = df["open_time"].parallel_apply(
            lambda x: pd.to_datetime(x, utc=True, unit = "ms")
            .tz_convert(tz)
        )
        df[f"close_t_{suffix}"] = df["close_time"].parallel_apply(
            lambda x: pd.to_datetime(x, utc=True, unit = "ms")
            .tz_convert(tz)
        )
        df["TradeDate"] = df[f"open_t_{suffix}"]\
            .parallel_apply(lambda x: int(x.date().strftime("%Y%m%d")))
        
        df.drop(columns = ["open_time", "close_time", "ignore"], inplace = True)
        df.sort_values(by = f"open_t_{suffix}", inplace = True)
        for col in df.select_dtypes(include='object').columns:
            df[col] = pd.to_numeric(df[col], errors = "ignore")

        df["vwap"] = df["quote_asset_volume"] / df["volume"]
            
        return df

    def get(self, freq: str, symbol: str, 
            end: pd.Timestamp, n: int = 1000, ltype: str = "spot") -> pd.DataFrame:
        ltype = HistoricalKlinesType.SPOT if ltype == "spot" else HistoricalKlinesType.FUTURES
        end = end.tz_convert(self.tz)
        try:
            klines = self.client.get_historical_klines(
                symbol = symbol, interval = freq, 
                end_str = str(end), 
                limit = n, klines_type = ltype,
            )
            kline_df = self.kline_converter(klines, self.tz)
            kline_df["symbol"] = symbol
            return kline_df
        
        except:
            logger.exception("Error")
            return pd.DataFrame()

if __name__ == "__main__":
    logger.add(f"{PROJ_DIR}/logger/KLineDownloader.log", rotation="1 day", retention="7 days")
    save_path = f"{PROJ_DIR}/data/bigdata/bn/10min/BTCUSDT_spot"
    if not os.path.exists(save_path): os.mkdir(save_path)

    bd = BNKLineDownloader()
    start = pd.Timestamp("2023-08-01").tz_localize(bd.tz)
    end = pd.Timestamp("2023-12-26").tz_localize(bd.tz)

    record = pd.DataFrame()
    while start < end:
        try:
            kline_df = bd.get("15m", "BTCUSDT", end, 1000, "spot")
            record = pd.concat([kline_df, record], axis = 0, ignore_index = True)
            logger.success(f"Successfully Get {end}")
            end = kline_df[f"open_t_hk"].min()
            time.sleep(0.2)
        except:
            logger.exception("Error")
            time.sleep(0.2)
            continue
    record.drop_duplicates(subset = ["open_t_hk"], inplace = True)
    record.reset_index(drop = True, inplace = True)
    # for date, group in record.groupby("TradeDate"):
    #     group.to_parquet(f"{save_path}/{date}.parquet")
    record.to_parquet(f"{save_path}/BTCUSDT.parquet")
# %%
