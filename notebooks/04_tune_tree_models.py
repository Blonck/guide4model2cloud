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

    clf_name = trial.suggest_categorical("classifier", ["rf", "xt", "gb"])

    if clf_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=trial.suggest_categorical(
                "rf_nest", [50, 100, 150, 200, 300, 500]
            ),
            max_depth=trial.suggest_int("rf_max_depth", 3, 15),
            min_samples_split=trial.suggest_int("rf_min_split", 2, 5),
            min_samples_leaf=trial.suggest_int("rf_min_leaf", 1, 8),
            bootstrap=trial.suggest_categorical("rf_bootstrap", [True, False]),
            class_weight=trial.suggest_categorical(
                "rf_cl_weight", ["balanced", "balanced_subsample"]
            ),
        )
    elif clf_name == "xt":
        clf = ExtraTreesClassifier(
            n_estimators=trial.suggest_categorical(
                "xt_nest", [50, 100, 150, 200, 300, 500]
            ),
            max_depth=trial.suggest_int("xt_max_depth", 3, 15),
            min_samples_split=trial.suggest_int("xt_min_split", 2, 5),
            min_samples_leaf=trial.suggest_int("xt_min_leaf", 1, 8),
            bootstrap=trial.suggest_categorical("xt_bootstrap", [True, False]),
            class_weight=trial.suggest_categorical(
                "xt_cl_weight", ["balanced", "balanced_subsample"]
            ),
        )
    elif clf_name == "gb":
        clf = GradientBoostingClassifier(
            learning_rate=trial.suggest_float("gb_lr", 0.01, 0.3),
            subsample=trial.suggest_float("gb_subsample", 0.5, 1.0),
            n_estimators=trial.suggest_categorical(
                "gb_nest", [50, 100, 150, 200, 300, 500]
            ),
            max_depth=trial.suggest_int("gb_max_depth", 2, 7),
            min_samples_split=trial.suggest_int("gb_min_split", 2, 5),
            min_samples_leaf=trial.suggest_int("gb_min_leaf", 1, 8),
            n_iter_no_change=20,
        )

    # k-fold over 5 split
    # on every split the average precision is calculated and optuna
    # can decide to prune this trial
    maes = []
    i = 0
    kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    for train_idx, val_idx in kf.split(x, y):
        x_train = x.loc[train_idx]
        y_train = y.loc[train_idx]

        x_train = enc.fit_transform(x_train, y_train)
        clf.fit(x_train, y_train)

        x_val = x.loc[val_idx]
        x_val = enc.transform(x_val)
        y_val_prob = clf.predict_proba(x_val)[:, 1]
        y_val_true = y.loc[val_idx]

        avg_pre = me.average_precision_score(y_val_true, y_val_prob)
        maes.append(avg_pre)

        # after three folds, allow optuna to prune
        if i >= 3:
            trial.report(np.mean(maes), i)
            if trial.should_prune():
                raise opt.exceptions.TrialPruned()
        i += 1

    return np.mean(maes)


# %%
jbfile = Path('study-tree-models.joblib')

if jbfile.exists():
    study = joblib.load(jbfile)
else:
    study = opt.create_study(pruner=opt.pruners.MedianPruner(), direction='maximize')

# %%
study.optimize(objective, timeout=60*5)

# %%
joblib.dump(study, 'study-tree-models.joblib')

# %%
opt.visualization.plot_optimization_history(study)

# %%
