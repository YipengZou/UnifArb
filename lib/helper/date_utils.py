import os

import pandas as pd


__location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))


def std_format(day):
    if '-' in day:
        y,m,d = day.split('-')
        res = y + m + d
    elif '/' in day:
        m,d,y = day.split('/')
        res = y + m + d
    else:
        res = day
    return res

def hyphen_format(day):
    day = std_format(day)
    return day[0:4] + '-' + day[4:6] + '-' + day[6:8]

def load_calendar():
    path = os.path.join(__location__, '..', '..', 'metadata', 'calendar.csv')
    return pd.read_csv(path, header=0, index_col=0).index.map(str).tolist()

def dateRange(start_date, end_date, sample='ALL'):
    calendar = load_calendar()
    assert start_date >= calendar[0] and end_date <= calendar[-1], 'invalid date range'
    res = [day for day in calendar if day >= start_date and day <= end_date]
    assert sample in ['ALL', 'INS', 'OOS']
    if sample == 'ALL':
        return res
    elif sample == 'INS':
        return [day for day in res if int(day[4:6])%2 == 1]
    elif sample == 'OOS':
        return [day for day in res if int(day[4:6])%2 == 0]

def toBusDate(day, shift = 0):
    days = load_calendar()
    day = std_format(day)
    if day in days:
        idx = days.index(day)
        idx += shift; assert idx >= 0, 'out of range'
        return days[idx]
    for idx in range(len(days)):
        if days[idx] > day:
            break
    assert days[idx] > day, 'out of range'
    idx -= 1
    idx += shift; assert idx >=0 and idx, 'out of range'
    return days[idx]
