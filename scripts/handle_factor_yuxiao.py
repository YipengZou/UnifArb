#%%
import pandas as pd
import os

factor_folder = "/home/zouyipeng/Workspace/UnifiedArb/metadata/PERFORMANCE_SUMMARY"
folders = os.listdir(factor_folder)
record_lst = []
for folder in folders:
    if folder == "ALL_FACTOR_SUMMARY":
        continue
    path = os.path.join(factor_folder, folder + "/raw_return_daily.csv")
    df = pd.read_csv(path)  # type: ignore
    ls_return = df.eval("p1 - p10")
    if ls_return.mean() < 0:
        ls_return *= -1
    ls_return.index = df["trade_date"]
    ls_return.index.name = "date"
    ls_return.name = folder
    ls_return.cumsum().plot()
    record_lst.append(ls_return)

#%%
pd.concat(record_lst, axis=1).to_csv("/home/zouyipeng/Workspace/UnifiedArb/metadata/CN_daily_return.csv")
# %%
