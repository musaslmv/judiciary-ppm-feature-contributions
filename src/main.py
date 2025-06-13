#!/usr/bin/env python3

from pathlib import Path
from src.trace_vectorizer import create_trace_vectors

if __name__ == "__main__":
    # assume this file lives in src/, repo_root is one level up
    repo_root    = Path(__file__).resolve().parent.parent
    data_dir     = repo_root / "data"

    completed_pkl = data_dir / "source_completed.pkl"
    active_pkl    = data_dir / "active_cases.pkl"
    out_pkl       = repo_root / "trace_vectors.pkl"

    # build and save your trace vectors
    tv = create_trace_vectors(completed_pkl, active_pkl)
    tv.to_pickle(out_pkl)
    print(f"Trace vectors written to {out_pkl}")
