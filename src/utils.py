import pandas as pd
import numpy as np

"""
Methods used for data cleanup
"""


def clean_na(df, columns, imp_type, na_value=None, shift_size=24):
    """
    Method to clean and impute time series data
    :param df: panda df to be cleaned
    :param columns: columns to be cleaned
    :param imp_type: imputation type (mean, median, etc.)
    :param na_value: value used to denote na values (e.g. -999.000)
    :param shift_size: what value should be imputed (from 24h ago, from 1h ago, etc.)
    :return: cleaned df
    """
    df.loc[:, 'date'] = pd.to_datetime(df.loc[:, 'date'], format='%Y-%m-%d %H:%M')
    if na_value is not None:
        df = df.replace(na_value, np.nan)
    df['day_date'] = pd.to_datetime(df['date']).dt.date
    for col in columns:
        df[col] = df[col].fillna(df.groupby('day_date')[col].transform(imp_type))
        while df[col].isna().sum():
            df[col] = df[col].fillna(df[col].shift(shift_size))
    df = df.set_index('date')
    return df


def add_day_trig(df):
    """
    Method to add day time feature
    :param df: df to be updated
    :return: df with trigonometric day time feature added
    """
    df["hour"] = [x.hour for x in df.index]
    df["day_cos"] = [np.cos(x * (2 * np.pi / 24)) for x in df["hour"]]
    df["day_sin"] = [np.sin(x * (2 * np.pi / 24)) for x in df["hour"]]
    return df


def add_month_trig(df):
    """
    Method to add extra time feature representing month
    :param df: df to be updated
    :return: df with trigonometric month time feature added
    """
    df["timestamp"] = [x.timestamp() for x in df.index]

    # Seconds in day
    s = 24 * 60 * 60
    # Seconds in year
    year = (365.25) * s
    df["month_cos"] = [np.cos((x) * (2 * np.pi / year)) for x in df["timestamp"]]
    df["month_sin"] = [np.sin((x) * (2 * np.pi / year)) for x in df["timestamp"]]
    return df
