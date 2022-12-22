import pandas as pd

def int_to_date(actual_start_date, t):
    date = actual_start_date + pd.Timedelta(t, unit='D')
    return date