# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import joblib
from pathlib import Path

import pandas as pd
import optuna as opt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
import sklearn.metrics as me


# ignore future warnings from numpy due to category_encoders
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

import category_encoders as ce

# %run ../app/preprocess.py

# %%
data = pd.read_parquet('../data/bank-additional-full.parquet')

train, test = train_test_split(data, test_size=0.15, random_state=42)

# %%
x = train.drop(columns=['y']).reset_index(drop=True)
y = train['y'].reset_index(drop=True)

x_test = test.drop(columns=['y']).reset_index(drop=True)
y_test = test['y'].reset_index(drop=True)

# %%
encoders = {
    "one-hot": ce.OneHotEncoder(
        drop_invariant=True, return_df=True, use_cat_names=True
    ),
    "woe": ce.WOEEncoder(drop_invariant=True, return_df=True),
    "binary": ce.BinaryEncoder(drop_invariant=True, return_df=True),
}


def objective(trial: opt.Trial):
    # only test dropping sozio economic facotrs
    drop_sozioeco = trial.suggest_categorical("drop_eco", [True, False])
    # rest of preprocessing keeps default values

    # categrorial encoding, try identical encoders for all columns (for now)
    enc_name = trial.suggest_categorical("encoder", ["one-hot", "woe", "binary"])
    enc = encoders[enc_name]

    x_tr = enc.fit_transform(x, y)

    param = {
        "verbosity": 0,
        "obective": "binary:logistic",
        "eval_metric": ["aucpr"],
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "booster": "gbtree",
        "lambda": trial.suggest_float("lambda", 1e-7, 0.5, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 0.5, log=True),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "eta": trial.suggest_loguniform("lr", 1e-5, 0.2),
        "gamma": trial.suggest_loguniform("gamma", 1e-8, 1.0),
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        )
    }
    
    dtrain = xgb.DMatrix(x_tr, label=y)
    cb = optuna.integration.XGBoostPruningCallback(trial, observation_key='test-aucpr')

    scores = xgb.cv(
        param,
        dtrain,
        nfold=5,
        stratified=True
    )
    
    test_aucpr = score['test-aucpr-mean'].values[-1]
    
    return test_aucpr


# %%
pruner = opt.pruners.MedianPruner(n_warmup_steps=40, interval_steps=4)

# %%
study = opt.create_study(direction='maximize', pruner=pruner)

# %%
study.optimize(objective, timeout=60*60*12, n_jobs=4)
