# %%
from datetime import datetime, timedelta
import pandas as pd
import calendar
def adjust_time(row: pd.Series) -> pd.Series:
    dt_str = row['Date-Time'][:19]  # Seconds
    dt_obj = datetime.strptime(dt_str, '%Y-%m-%dT%H:%M:%S')
    dt_obj += timedelta(hours=row['GMT Offset'])
    return dt_obj

def get_nth_day(year, month, nth, from_end = False):
    _, num_days = calendar.monthrange(year, month)
    
    if from_end:
        day = num_days - (nth - 1)
    else:
        day = nth
    
    return datetime(year, month, day).strftime("%Y-%m-%d")

def cme_time_trans(df: pd.DataFrame) -> pd.DataFrame:
    df['Date-Time'] = df.apply(adjust_time, axis=1)
    shanghai_tz = "Asia/Shanghai"
    df['Date-Time'] = pd.to_datetime(df['Date-Time']).dt.tz_localize('UTC').dt.tz_convert(shanghai_tz)
    return df

def convert_time_local(t: str, tz: str = "Asia/Shanghai") -> pd.Timestamp:
    return pd.to_datetime(t).tz_localize(tz)

def get_cme_main(prev_n = 5):
    trans_date = {
        "1GC0222.csv": (get_nth_day(2022, 1, 1), get_nth_day(2022, 2, prev_n, from_end = True)),
        "1GC0422.csv": (get_nth_day(2022, 2, prev_n, from_end = True), get_nth_day(2022, 4, prev_n, from_end = True)),
        "1GC0622.csv": (get_nth_day(2022, 4, prev_n, from_end = True), get_nth_day(2022, 6, prev_n, from_end = True)),
        "1GC0822.csv": (get_nth_day(2022, 6, prev_n, from_end = True), get_nth_day(2022, 8, prev_n, from_end = True)),
        "1GC1022.csv": (get_nth_day(2022, 8, prev_n, from_end = True), get_nth_day(2022, 10, prev_n, from_end = True)),
        "1GC1222.csv": (get_nth_day(2022, 10, prev_n, from_end = True), get_nth_day(2022, 12, prev_n, from_end = True)),
        "1GC0223.csv": (get_nth_day(2022, 12, prev_n, from_end = True), get_nth_day(2023, 2, prev_n, from_end = True)),
        "1GC0423.csv": (get_nth_day(2023, 2, prev_n, from_end = True), get_nth_day(2023, 4, prev_n, from_end = True)),
        "1GC0623.csv": (get_nth_day(2023, 4, prev_n, from_end = True), get_nth_day(2023, 6, prev_n, from_end = True)),
        "1GC0823.csv": (get_nth_day(2023, 6, prev_n, from_end = True), get_nth_day(2023, 8, prev_n, from_end = True)),
        "1GC1023.csv": (get_nth_day(2023, 8, prev_n, from_end = True), get_nth_day(2023, 10, prev_n, from_end = True)),
        "1GC1223.csv": (get_nth_day(2023, 10, prev_n, from_end = True), get_nth_day(2023, 12, prev_n, from_end = True)),
    }
    cme_record = []
    cme_folder = "/home/ubuntu/data/gold/"
    for file in trans_date.keys():
        df = pd.read_csv(f"{cme_folder}/{file}", index_col = 0)
        df = cme_time_trans(df)
        print(df.tail(5))
        cme_record.append(
            df.loc[df["Date-Time"].between(
            convert_time_local(trans_date[file][0]),
            convert_time_local(trans_date[file][1]),
            )])
        print(trans_date[file])
        print(f"Finish {file}")
        
    cme_record = pd.concat(cme_record).rename(columns = {"Date-Time": "time"})
    cme_record.reset_index(drop = True)\
        .sort_values(by = "time")\
            .to_parquet("/home/ubuntu/data/gold/continue_main/cme_main_prev15.parquet")

#%%
def get_shf_main(prev_n = 5):
    df = pd.read_csv("/home/ubuntu/data/gold/au_future.csv", index_col = 0)
    df["time"] = pd.to_datetime(df['index']).dt.tz_localize("Asia/Shanghai")
    trans_date = {
            "AU2202.SHF": (get_nth_day(2022, 1, 1), get_nth_day(2022, 1, prev_n, from_end = True)),
            "AU2204.SHF": (get_nth_day(2022, 1, prev_n, from_end = True), get_nth_day(2022, 3, prev_n, from_end = True)),
            "AU2206.SHF": (get_nth_day(2022, 3, prev_n, from_end = True), get_nth_day(2022, 5, prev_n, from_end = True)),
            "AU2208.SHF": (get_nth_day(2022, 5, prev_n, from_end = True), get_nth_day(2022, 7, prev_n, from_end = True)),
            "AU2210.SHF": (get_nth_day(2022, 7, prev_n, from_end = True), get_nth_day(2022, 9, prev_n, from_end = True)),
            "AU2212.SHF": (get_nth_day(2022, 9, prev_n, from_end = True), get_nth_day(2022, 11, prev_n, from_end = True)),
            "AU2302.SHF": (get_nth_day(2022, 11, prev_n, from_end = True), get_nth_day(2023, 1, prev_n, from_end = True)),
            "AU2304.SHF": (get_nth_day(2023, 1, prev_n, from_end = True), get_nth_day(2023, 3, prev_n, from_end = True)),
            "AU2306.SHF": (get_nth_day(2023, 3, prev_n, from_end = True), get_nth_day(2023, 5, prev_n, from_end = True)),
            "AU2308.SHF": (get_nth_day(2023, 5, prev_n, from_end = True), get_nth_day(2023, 7, prev_n, from_end = True)),
            "AU2310.SHF": (get_nth_day(2023, 7, prev_n, from_end = True), get_nth_day(2023, 9, prev_n, from_end = True)),
            "AU2312.SHF": (get_nth_day(2023, 9, prev_n, from_end = True), get_nth_day(2023, 11, prev_n, from_end = True)),
        }

    shf_record = []
    for contract, (st, ed) in trans_date.items():
        print(contract, st, ed)
        df_use = df.loc[df["windcode"] == contract]
        shf_record.append(
            df_use.loc[df_use["time"].between(
                convert_time_local(st),
                convert_time_local(ed),
            )])
    pd.concat(shf_record).reset_index(drop = True).to_parquet("/home/ubuntu/data/gold/continue_main/shf_main_prev15.parquet")

# %%
# get_cme_main(prev_n = 15)
get_shf_main(prev_n = 15)
