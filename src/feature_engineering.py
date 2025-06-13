import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from collections import defaultdict

def build_cooccurrence_matrix(log_df: pd.DataFrame) -> (np.ndarray, list):
    traces = defaultdict(list)
    for case_id, group in log_df.sort_values('Event date').groupby('Case ID'):
        traces[case_id] = group.sort_values('Event date')['Event ID'].tolist()
    cooccurrence = defaultdict(lambda: defaultdict(int))
    for trace in traces.values():
        for i in range(len(trace)-1):
            a, b = trace[i], trace[i+1]
            cooccurrence[a][b] += 1
            cooccurrence[b][a] += 1
    events = sorted(set(e for trace in traces.values() for e in trace))
    idx_map = {e:i for i,e in enumerate(events)}
    n = len(events)
    matrix = np.zeros((n,n), dtype=int)
    for e1, row in cooccurrence.items():
        i = idx_map[e1]
        for e2, count in row.items():
            j = idx_map[e2]
            matrix[i,j] = count
    return matrix, events

def cluster_events(co_mat: np.ndarray, events: list, n_clusters: int=10, random_state: int=42) -> dict:
    mat_norm = normalize(co_mat, norm='l2', axis=1)
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = km.fit_predict(mat_norm)
    return {event: str(label) for event, label in zip(events, labels)}

def encode_event_counts(df: pd.DataFrame, case_id_col: str='Case ID', event_col: str='Event ID', top_n: int=50) -> pd.DataFrame:
    event_counts = df[event_col].value_counts()
    top_events = event_counts.head(top_n).index.tolist()
    result = df[[case_id_col, event_col]].copy()
    result['seq_num'] = result.groupby(case_id_col).cumcount()
    dummies = pd.get_dummies(result[event_col]).reindex(columns=top_events, fill_value=0)
    dummies.columns = [f"{event_col}_{e}" for e in dummies.columns]
    result = pd.concat([result, dummies], axis=1)
    for e in top_events:
        col = f"{event_col}_{e}"
        result[col] = result.groupby(case_id_col)[col].cumsum()
    return result

def encode_cluster_counts(df: pd.DataFrame, case_id_col: str='Case ID', n_clusters: int=10, cluster_col: str='Cluster') -> pd.DataFrame:
    result = df[[case_id_col, cluster_col]].copy()
    result['seq_num'] = result.groupby(case_id_col).cumcount()
    dummies = pd.get_dummies(result[cluster_col], prefix=cluster_col)
    for i in range(n_clusters):
        col = f"{cluster_col}_{i}"
        if col not in dummies.columns:
            dummies[col] = 0
    dummies = dummies.reindex(columns=[f"{cluster_col}_{i}" for i in range(n_clusters)], fill_value=0)
    result = pd.concat([result, dummies], axis=1)
    for i in range(n_clusters):
        col = f"{cluster_col}_{i}"
        result[col] = result.groupby(case_id_col)[col].cumsum()
    return result

def encode_elapsed_time(df: pd.DataFrame, case_id_col: str='Case ID', timestamp_col: str='Event date') -> pd.DataFrame:
    temp = df[[case_id_col, timestamp_col]].copy()
    temp[timestamp_col] = pd.to_datetime(temp[timestamp_col])
    temp['seq_num'] = temp.groupby(case_id_col).cumcount()
    temp = temp.sort_values([case_id_col, timestamp_col, 'seq_num'])
    temp['Elapsed_time'] = temp.groupby(case_id_col)[timestamp_col].transform(lambda x: (x - x.iloc[0]).dt.days)
    temp['Time_since_last_event'] = temp.groupby(case_id_col)[timestamp_col].diff().dt.days.fillna(0)
    return temp[[case_id_col, 'seq_num', 'Elapsed_time', 'Time_since_last_event']]

def encode_time_features(df: pd.DataFrame, case_id_col: str='Case ID', timestamp_col: str='Event date') -> pd.DataFrame:
    temp = df[[case_id_col, timestamp_col]].copy()
    temp[timestamp_col] = pd.to_datetime(temp[timestamp_col])
    temp['seq_num'] = temp.groupby(case_id_col).cumcount()
    temp = temp.sort_values([case_id_col, timestamp_col, 'seq_num'])
    temp['Weekday'] = temp[timestamp_col].dt.dayofweek
    temp['Week_number'] = temp[timestamp_col].dt.isocalendar().week
    temp['Month_number'] = temp[timestamp_col].dt.month
    return temp[[case_id_col, 'seq_num', 'Weekday', 'Week_number', 'Month_number']]

def encode_state_ids(df: pd.DataFrame, case_id_col: str='Case ID', state_id_col: str='State ID', common_state_ids: list=['AS','GC','UT','PC','D1','D2']) -> pd.DataFrame:
    temp = df[[case_id_col, state_id_col]].copy()
    temp['seq_num'] = temp.groupby(case_id_col).cumcount()
    for s in common_state_ids:
        temp[f"{state_id_col}_{s}"] = (temp[state_id_col] == s).astype(int)
    temp = temp.sort_values([case_id_col, 'seq_num'])
    cols = [f"{state_id_col}_{s}" for s in common_state_ids]
    return temp[[case_id_col, 'seq_num'] + cols]

def encode_state_elapsed_time(df: pd.DataFrame, case_id_col: str='Case ID', state_id_col: str='State ID', timestamp_col: str='Event date', common_state_ids: list=['AS','GC','UT','PC','D1','D2']) -> pd.DataFrame:
    temp = df[[case_id_col, state_id_col, timestamp_col]].copy()
    temp[timestamp_col] = pd.to_datetime(temp[timestamp_col])
    temp['seq_num'] = temp.groupby(case_id_col).cumcount()
    temp = temp.sort_values([case_id_col, 'seq_num'])
    for s in common_state_ids:
        temp[f"{state_id_col}_{s}_elapsed_time"] = 0
    for case, group in temp.groupby(case_id_col):
        last_time = {}
        for idx, row in group.iterrows():
            cur_state = row[state_id_col]
            cur_time = row[timestamp_col]
            for s in common_state_ids:
                if cur_state == s:
                    if s in last_time:
                        temp.loc[idx, f"{state_id_col}_{s}_elapsed_time"] = (cur_time - last_time[s]).days
                    else:
                        temp.loc[idx, f"{state_id_col}_{s}_elapsed_time"] = 0
                else:
                    temp.loc[idx, f"{state_id_col}_{s}_elapsed_time"] = 0
            last_time[cur_state] = cur_time
    cols = [f"{state_id_col}_{s}_elapsed_time" for s in common_state_ids]
    return temp[[case_id_col, 'seq_num'] + cols]

def encode_case_attributes(df: pd.DataFrame, case_id_col: str='Case ID', judge_id_col: str='Judge ID', category_col: str='Case category', section_col: str='Case Section') -> pd.DataFrame:
    temp = df[[case_id_col, judge_id_col, category_col, section_col]].copy()
    temp['seq_num'] = temp.groupby(case_id_col).cumcount()
    return temp[[case_id_col, 'seq_num', judge_id_col, category_col, section_col]]

def encode_remaining_time(df: pd.DataFrame, case_id_col: str='Case ID', timestamp_col: str='Event date') -> pd.DataFrame:
    temp = df[[case_id_col, timestamp_col]].copy()
    temp[timestamp_col] = pd.to_datetime(temp[timestamp_col])
    temp['seq_num'] = temp.groupby(case_id_col).cumcount()
    temp = temp.sort_values([case_id_col, timestamp_col, 'seq_num'])
    temp['Remaining_time'] = temp.groupby(case_id_col)[timestamp_col].transform(lambda x: (x.iloc[-1] - x).dt.days)
    return temp[[case_id_col, 'seq_num', 'Remaining_time']]

def encode_last_two_event_ids(df: pd.DataFrame, case_id_col: str='Case ID', event_col: str='Event ID') -> pd.DataFrame:
    temp = df[[case_id_col, event_col]].copy()
    temp['seq_num'] = temp.groupby(case_id_col).cumcount()
    last = []
    second_last = []
    for case, group in temp.groupby(case_id_col):
        evs = group[event_col].tolist()
        for i, ev in enumerate(evs):
            last.append(ev)
            second_last.append(evs[i-1] if i > 0 else None)
    temp['Last_event_ID'] = last
    temp['Second_last_event_ID'] = second_last
    return temp[[case_id_col, 'seq_num', 'Last_event_ID', 'Second_last_event_ID']]

def encode_judge_change_features(df: pd.DataFrame, case_id_col: str='Case ID', judge_id_col: str='Judge ID') -> pd.DataFrame:
    temp = df[[case_id_col, judge_id_col]].copy()
    temp['seq_num'] = temp.groupby(case_id_col).cumcount()
    temp = temp.sort_values([case_id_col, 'seq_num'])
    temp['judge_changed'] = temp.groupby(case_id_col)[judge_id_col].apply(lambda x: (x != x.shift(1)).fillna(False).astype(int))
    return temp[[case_id_col, 'seq_num', 'judge_changed']]

def merge_feature_dfs(df_list: list, case_id_col: str='Case ID') -> pd.DataFrame:
    from functools import reduce
    df_merged = reduce(lambda l, r: pd.merge(l, r, on=[case_id_col, 'seq_num'], how='left'), df_list)
    if 'seq_num' in df_merged.columns:
        df_merged = df_merged.drop(columns=['seq_num'])
    return df_merged

def merge_active_cases(trace_df: pd.DataFrame, active_cases_df: pd.DataFrame) -> pd.DataFrame:
    df = trace_df.merge(active_cases_df, left_on='Event date', right_on='Date', how='left')
    df = df.drop(columns=['Date'])
    df['Total_Active_Cases'] = df['Total_Active_Cases'].fillna(0).astype(int)
    return df

def add_judge_workload(df: pd.DataFrame) -> pd.DataFrame:
    case_judge = df[['Case ID','Judge ID']].drop_duplicates().set_index('Case ID')['Judge ID'].to_dict()
    workloads = []
    for _, row in df.iterrows():
        act_dict = row.get('act_Cases_dict', {})
        if isinstance(act_dict, str):
            try: active_cases = eval(act_dict)
            except: active_cases = {}
        else:
            active_cases = act_dict or {}
        count = sum(1 for case in active_cases if case_judge.get(case) == row['Judge ID'])
        workloads.append(count)
    df['Judge Workload'] = workloads
    return df
