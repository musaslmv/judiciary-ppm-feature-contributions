import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

def split_trace_vectors(data: pd.DataFrame,
                        log_df: pd.DataFrame = None,
                        train_ratio: float = 0.8,
                        forbidden_states: list = None):
    if forbidden_states is None:
        forbidden_states = ['DF','AK','CL','PH','AN','IA','IC','IM','IT','ZM'] # states where the cases are ended

    data = data.copy()
    data['Event date'] = pd.to_datetime(data['Event date'])
    data = data.sort_values(['Event date','Event ID'])

    # filter out any cases ending on/after 2024-01-01
    if log_df is not None:
        log = log_df.copy()
        log['Event date'] = pd.to_datetime(log['Event date'])
        bounds = (log
                  .groupby('Case ID')['Event date']
                  .agg(start_ts='min', end_ts='max')
                  .reset_index())
        valid = bounds[bounds['end_ts'] < pd.Timestamp("2024-01-01")]['Case ID']
        data = data[data['Case ID'].isin(valid)]
        log  = log[log['Case ID'].isin(valid)]
    else:
        log = None

    case_bounds = (data.groupby('Case ID')['Event date']
                   .agg(start_ts='min', end_ts='max')
                   .reset_index()
                   .sort_values('start_ts'))
    split_index = int(train_ratio * len(case_bounds))
    split_ts = case_bounds['start_ts'].iloc[split_index]

    train_ids = case_bounds[case_bounds['end_ts'] < split_ts]['Case ID']
    test_ids  = case_bounds[case_bounds['start_ts'] > split_ts]['Case ID']

    train = data[data['Case ID'].isin(train_ids)].sort_values(['Event date','Event ID'])
    test  = data[data['Case ID'].isin(test_ids )].sort_values(['Event date','Event ID'])

    if log is not None and forbidden_states:
        forbidden = (log[log['State ID'].isin(forbidden_states)]
                     [['Case ID','Event ID','Event date']]
                     .drop_duplicates())
        forbidden_set = set(map(tuple, forbidden.to_records(index=False)))
        test = test[~test[['Case ID','Event ID','Event date']]
                    .apply(tuple, axis=1)
                    .isin(forbidden_set)]

    return train, test

def train_and_evaluate(trace_vectors: pd.DataFrame,
                       log: pd.DataFrame,
                       features_to_exclude: list = None,
                       use_gpu: bool = True,
                       hyperparams: dict = None,
                       optimize_hyperparams: bool = False) -> dict:
    # split
    train_df, test_df = split_trace_vectors(trace_vectors, log_df=log)

    # drop cols
    base_drop = ['Case ID','Event date','Remaining_time']
    drop_cols = base_drop + (features_to_exclude or [])

    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df['Remaining_time']
    X_test  = test_df .drop(columns=drop_cols)
    y_test  = test_df ['Remaining_time']

    # identify and cast categoricals
    possible_cat = ['Event ID','State ID','Judge ID','Case category',
                    'Case Section','Last_event_ID','Second_last_event_ID','judge_changed']
    cat_features = [c for c in possible_cat if c in X_train.columns]
    for c in cat_features:
        X_train[c] = X_train[c].astype(str)
        X_test[c]  = X_test[c].astype(str)

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    test_pool  = Pool(X_test,  y_test,  cat_features=cat_features)

    params = {
        'iterations':     1000,
        'learning_rate':  0.1,
        'depth':          6,
        'task_type':      'GPU' if use_gpu else 'CPU',
        'devices':        '0' if use_gpu else None,
        'random_seed':    42,
        'verbose':        100
    }
    if hyperparams:
        params.update(hyperparams)

    # optional hyperopt tuning
    if optimize_hyperparams:
        X_tune, X_val, y_tune, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42)
        cat_tune = [c for c in cat_features if c in X_tune.columns]
        for c in cat_tune:
            X_tune[c] = X_tune[c].astype(str)
            X_val[c]  = X_val[c].astype(str)

        def objective(p):
            p.update({
                'task_type': params['task_type'],
                'devices':   params['devices'],
                'random_seed': 42,
                'verbose': 0
            })
            m = CatBoostRegressor(**p)
            m.fit(X_tune, y_tune, cat_features=cat_tune, verbose=0)
            preds = m.predict(X_val)
            return {'loss': mean_absolute_error(y_val, preds), 'status': STATUS_OK}

        space = {
            'depth':        hp.choice('depth', [4,6,8,10]),
            'learning_rate':hp.uniform('lr', 0.01, 0.3),
            'l2_leaf_reg':  hp.uniform('l2', 1, 10),
            'iterations':   hp.choice('iters', [100,300,500])
        }
        best = fmin(objective, space, algo=tpe.suggest, max_evals=20, trials=Trials())
        params.update({
            'depth':       [4,6,8,10][best['depth']],
            'iterations':  [100,300,500][best['iters']],
            'learning_rate': best['lr']
        })

    # train final
    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=200)

    preds = model.predict(test_pool)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)

    fi = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.get_feature_importance(train_pool)
    }).sort_values('Importance', ascending=False)

    results_df = pd.DataFrame({
        'Case ID': test_df['Case ID'],
        'Event date': test_df['Event date'],
        'Actual Remaining_time': y_test,
        'Predicted Remaining_time': preds
    })

    return {
        'model':              model,
        'rmse':               rmse,
        'mae':                mae,
        'feature_importance': fi,
        'results_df':         results_df,
        'num_train_cases':    train_df['Case ID'].nunique(),
        'num_test_cases':     test_df['Case ID'].nunique()
    }
