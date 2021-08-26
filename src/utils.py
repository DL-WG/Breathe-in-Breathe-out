from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.linalg import inv



def clean_na(df, columns, imp_type, na_value=None, shift_size=24):
    df.loc[:, 'date'] = pd.to_datetime(df.loc[:, 'date'], format='%Y-%m-%d %H:%M')
    if na_value is not None:
        df = df.replace(na_value, np.nan)  # (-999.0000, np.nan)
    df['day_date'] = pd.to_datetime(df['date']).dt.date
    for col in columns:
        df[col] = df[col].fillna(df.groupby('day_date')[col].transform(imp_type))
        while df[col].isna().sum():
            df[col] = df[col].fillna(df[col].shift(shift_size))
    df = df.set_index('date')
    return df


def add_day_trig(df):
    df["hour"] = [x.hour for x in df.index]
    df["day_cos"] = [np.cos(x * (2 * np.pi / 24)) for x in df["hour"]]
    df["day_sin"] = [np.sin(x * (2 * np.pi / 24)) for x in df["hour"]]
    return df


def add_month_trig(df):
    df["timestamp"] = [x.timestamp() for x in df.index]

    # Seconds in day
    s = 24 * 60 * 60
    # Seconds in year
    year = (365.25) * s
    df["month_cos"] = [np.cos((x) * (2 * np.pi / year)) for x in df["timestamp"]]
    df["month_sin"] = [np.sin((x) * (2 * np.pi / year)) for x in df["timestamp"]]
    return df

