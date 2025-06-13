import pandas as pd

def sort_log(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(['Case ID', 'Event date']).reset_index(drop=True)

def filter_valid_judges(df: pd.DataFrame) -> pd.DataFrame:
    mask = df.groupby('Case ID')['Judge ID'].transform(lambda x: x.notnull().any())
    return df[mask].reset_index(drop=True)
