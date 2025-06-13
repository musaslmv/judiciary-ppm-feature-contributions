import pandas as pd
from pathlib import Path

def load_main_log(pickle_path: Path) -> pd.DataFrame:
    df = pd.read_pickle(pickle_path)
    df = df.rename(columns={
        'NUMPRO': 'Case ID',
        'NUMGIU': 'Judge ID',
        'DATAEV': 'Event date',
        'CODICEOGGETTO': 'Case category',
        'CDESCR': 'Event description',
        'CCODST': 'State ID',
        'CTIPSE': 'Case Section',
        'CCDOEV': 'Event ID'
    })
    required = ['Case ID','Event ID','Event date','Event description','State ID','Case Section','Judge ID','Case category']
    missing = set(required) - set(df.columns)
    if missing:
        raise KeyError(f"Missing required main log columns: {missing}")
    df['Event date'] = pd.to_datetime(df['Event date'])
    return df[required]

def load_active_cases(pickle_path: Path) -> pd.DataFrame:
    df = pd.read_pickle(pickle_path)
    base = ['Date', 'Total_Active_Cases', 'act_Cases_dict']
    missing = set(base) - set(df.columns)
    if missing:
        raise KeyError(f"Missing required active cases columns: {missing}")
    df['Date'] = pd.to_datetime(df['Date'])
    return df
