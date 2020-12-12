import pandas as pd
import os

def get_sensor_data(path, ascending=False):
    sensor_data = pd.read_csv(path).drop(
        ['Unnamed: 0'], axis=1).set_index('date')
    sensor_data.index = pd.to_datetime(sensor_data.index)
    sensor_data = sensor_data.sort_index(ascending=ascending)
    return sensor_data

def get_weekly_field(df, field, amount=30):
  return df[field].resample('W').mean().fillna('')[:amount]

def get_dates_from_series(series):
    return list(map(lambda x: str(x)[:10], list(series.index)))
