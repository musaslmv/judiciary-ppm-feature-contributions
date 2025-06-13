#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

from src.trace_vectorizer import create_trace_vectors
from src.data_loader       import load_main_log
from src.train_evaluate    import train_and_evaluate

def main():
    # repo_root/src/ → repo_root/data/
    repo_root    = Path(__file__).resolve().parent.parent
    data_dir     = repo_root / "data"

    completed_pkl = data_dir / "source_completed.pkl"
    active_pkl    = data_dir / "active_cases.pkl"
    out_csv       = repo_root / "experiment_results.csv"

    # 1) build trace vectors & load original log
    tv  = create_trace_vectors(completed_pkl, active_pkl)
    log = load_main_log(completed_pkl)

    # 2) define your feature‐sets exactly as in your table
    case_attrs   = ["Judge ID", "Case category", "Case Section"]
    states       = [c for c in tv.columns if c.startswith("State ID_")]
    temporal     = ["Elapsed_time", "Time_since_last_event", "Weekday", "Week_number", "Month_number"]
    intercase    = [c for c in tv.columns if "_actv_cases" in c] + (["Total_Active_Cases"] if "Total_Active_Cases" in tv.columns else [])
    cf_events    = [c for c in tv.columns if c.startswith("Event ID_")]
    cf_clusters  = [c for c in tv.columns if c.startswith("Cluster_")]
    cf_last      = ["Last_event_ID", "Second_last_event_ID"]
    judge_wl     = ["Judge Workload"] if "Judge Workload" in tv.columns else []
    all_feats    = [c for c in tv.columns if c not in ["Case ID","Event date","Remaining_time"]]

    FEATURE_SETS = {
        "control-flow-events":                    cf_events,
        "control-flow-clusters":                  cf_clusters,
        "control-flow-last-events":               cf_last,
        "states":                                 states,
        "categorical":                            case_attrs,
        "intercase":                              intercase,
        "temporal":                               temporal,
        "judge_workload":                         judge_wl,
        "control-flow-events+states":             cf_events + states,
        "control-flow-clusters+states":           cf_clusters + states,
        "control-flow-last-events+states":        cf_last + states,
        "categorical+temporal":                   case_attrs + temporal,
        "control-flow-events+states+temporal":    cf_events + states + temporal,
        "ALL_FEATURES":                           all_feats,
        "states+categorical+temporal":            states + case_attrs + temporal,
        "states+categorical+temporal+events":     states + case_attrs + temporal + cf_events,
        "states+categorical+temporal+last_events":states + case_attrs + temporal + cf_last,
        "states+categorical":                     states + case_attrs,
        "states+categorical+intercase":           states + case_attrs + intercase,
        "states+categorical+temporal+last_events+judge_workload":
                                                 states + case_attrs + temporal + cf_last + judge_wl,
        "states+categorical+temporal+clusters":   states + case_attrs + temporal + cf_clusters
    }

    # 3) run experiments
    results = []
    for name, feats in FEATURE_SETS.items():
        essentials = {"Case ID","Event date","Remaining_time"}
        keep       = set(feats) | essentials
        exclude    = [c for c in tv.columns if c not in keep]

        res = train_and_evaluate(
            trace_vectors=tv,
            log=log,
            features_to_exclude=exclude,
            use_gpu=True,
            optimize_hyperparams=False
        )

        top5 = res["feature_importance"].head(5)["Feature"].tolist()
        print(f"{name:45s} → MAE={res['mae']:.2f}, RMSE={res['rmse']:.2f}, Top5={top5}")

        results.append({
            "Feature Set":    name,
            "MAE":            res["mae"],
            "RMSE":           res["rmse"],
            "Top 5 Features": top5
        })

    # 4) save to CSV
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"\nExperiment results written to {out_csv}")

if __name__ == "__main__":
    main()
