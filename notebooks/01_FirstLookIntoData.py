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

# %% [markdown]
# # EDA
#
# The idea of this first exploratory data analysis is to do the minimum amount of work
# necessary to create a, probably underwhelming, but working model.

# %%
import pandas as pd
import numpy as np
from IPython.display import display

import plotly.express as px

# %%
data = pd.read_csv('../data/bank-additional-full.csv', sep=';')

# %%
data.info()

# %% [markdown]
# ## Target Variable

# %% [markdown]
# * target variable has two classes: no == `don't subscribe ` and yes == `subscribe`.
# * huge class imbalance ~0.2% fraud cases

# %%
data['y'].value_counts(normalize=True)

# %% [markdown]
# In binary classification tasks the target are usually denoted with 0 and 1.

# %%
data['y'] = data['y'].map({'no': 0, 'yes': 1})

# %% [markdown]
# ## NA's
#
# One must at least check if any of the fields contain NA data, which is not the case here.

# %%
data.isna().sum()

# %% [markdown]
# ## Categorial vairables
#
# Without going into the details of the categorial variables, we must at least check the number of different classes per categorical variable. In the case one of them has a very high count of different classes, this needs to be handled by manually grouping the more seldom classes togther.
#
# Luckely, that is not the case here.

# %%
cat_cols = data.dtypes[data.dtypes == 'object'].index

# %%
px.bar(data[cat_cols].nunique(), labels={'index': '', 'value': '#classes'}).show("png")

# %%
for col in cat_cols:
    print(col)
    display(data[col].value_counts())
