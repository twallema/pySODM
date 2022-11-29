import pandas as pd

def date_to_diff(actual_start_date, end_date):
    """
    Convert date string to int (i.e. number of days since day 0 of simulation,
    which is warmup days before actual_start_date)
    """
    return int((pd.to_datetime(end_date)-pd.to_datetime(actual_start_date))/pd.to_timedelta('1D'))

def int_to_date(actual_start_date, t):
    date = actual_start_date + pd.Timedelta(t, unit='D')
    return date