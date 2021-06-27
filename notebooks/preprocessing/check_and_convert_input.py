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
from pathlib import Path
import pandas as pd

# %run ../helper/git_root.py

# %%
dpath = Path(get_git_root('.')) / 'data'

# %%
data = pd.read_csv(dpath / 'bank-additional-full.csv', sep=';')

# %%
# check that the all expected columns are present
assert set(data.columns) == set(['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
                                 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
                                 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
                                 'cons.conf.idx', 'euribor3m', 'nr.employed', 'y'])

# drop duration columns, since it should not be used for building the model,
# see data information '../../data/bank-additional-names.txt'
data = data.drop(columns=['duration'])

# %%
data.info()

# %%
# convert strings in categorical variables
cat_cols = data.dtypes[data.dtypes == 'object'].index

for col in cat_cols:
    data[col] = data[col].astype('category')

# %%
# convert target in common representation
data['y'] = data['y'].map({'no': 0, 'yes': 1})

# %%
data.to_parquet(dpath / 'bank-additional-full.parquet')
