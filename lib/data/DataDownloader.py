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