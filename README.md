# Predicting Remaining Time for Legal Cases

This repository is supplementary for *Understanding Feature Contributions to Remaining Time Prediction in Judicial Processes*, a paper submitted to the BPM PLC 2025 workshop. We focus on five complementary feature categories: control-flow (activity sequences), temporal features, case attributes, process states, and judge workload—and conduct an ablation study to assess their individual and combined predictive contributions.

## Data

The dataset used in this study cannot be made publicly available due to confidentiality requirements imposed by data protection laws.

Place your preprocessed event logs in the `data/` directory (at the repo root):

* `source_completed.pkl`: completed cases with full event history
* `active_cases.pkl`: snapshot of daily active case counts

## Usage

1. **Generate Trace Vectors**

   ```bash
   python main.py
   ```

   Builds and saves a feature matrix (`trace_vectors.pkl`) combining sequence, temporal, and contextual features.

2. **Run Ablation Experiments**

   ```bash
   python experiments_steps.py
   ```

   Trains and evaluates CatBoost regression models on different feature subsets and records results in `experiment_results.csv`.

## Experiments and Findings

* **Feature Sets Evaluated**:  Control-flow, Temporal Context, Case Attributes, Process States, Judge Workload.
* **Ablation Study**: Sequentially adding feature categories to a baseline model to quantify their impact on mean absolute error (MAE).
* **Key Result**: Combining case attributes with process state and temporal context yields an MAE within an acceptable range for practical use, highlighting the importance of contextual information in predicting legal case durations.

## Prerequisites

Before running any of the scripts, provide two preprocessed pickle files in your `data/` folder:

1. **Completed cases log** (`source_completed.pkl`)
   A pandas DataFrame with **at least** these columns (post-rename):

   * `Case ID`            — original `NUMPRO`
   * `Event ID`           — original `CCDOEV`
   * `Event date`         — original `DATAEV` (converted to `datetime`)
   * `Event description`  — original `CDESCR`
   * `State ID`           — original `CCODST`
   * `Case Section`       — original `CTIPSE`
   * `Judge ID`           — original `NUMGIU`
   * `Case category`      — original `CODICEOGGETTO`

2. **Active cases snapshot** (`active_cases.pkl`)
   A pandas DataFrame with **at least**:

   * `Date`                — snapshot date (converted to `datetime`)
   * `Total_Active_Cases`  — total number of active cases per date
   * `act_Cases_dict`      — dict-like mapping `Case ID` → metadata (used to compute judge workload)
   * *Optional* columns for per-state/section active-case counts, e.g.:

     * `AS_actv_cases`, `GC_actv_cases`, …
     * `01_actv_cases`, `SectionX_actv_cases`, etc.

Please ensure that any terminal or forbidden states are appropriately managed by your input data (so that cases appear as resolved when they truly have ended).

## Dependencies

* Python 3.7+
* pandas
* numpy
* scikit-learn
* catboost
* hyperopt
