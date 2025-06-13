import pandas as pd
from pathlib import Path
from src.data_loader import load_main_log, load_active_cases
from src.preprocessing import sort_log, filter_valid_judges
import src.feature_engineering as fe

def create_trace_vectors(main_log_path: Path,
                         active_cases_path: Path,
                         n_clusters: int = 10) -> pd.DataFrame:
    log = load_main_log(main_log_path)
    active = load_active_cases(active_cases_path)

    log = sort_log(log)
    log = filter_valid_judges(log)

    co_mat, events = fe.build_cooccurrence_matrix(log)
    cluster_map = fe.cluster_events(co_mat, events, n_clusters=n_clusters)
    log['Cluster'] = log['Event ID'].map(cluster_map)

    df_list = [
        fe.encode_event_counts(log),
        fe.encode_cluster_counts(log, n_clusters=n_clusters),
        fe.encode_elapsed_time(log),
        fe.encode_time_features(log),
        fe.encode_state_ids(log),
        fe.encode_state_elapsed_time(log),
        fe.encode_case_attributes(log),
        fe.encode_remaining_time(log),
        fe.encode_last_two_event_ids(log),
    ]

    merged = fe.merge_feature_dfs(df_list)
    with_active = fe.merge_active_cases(merged, active)
    result = fe.add_judge_workload(with_active)
    return result
