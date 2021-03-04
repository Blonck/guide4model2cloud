# -*- coding: utf-8 -*-
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# %%
data = pd.read_parquet('../data/bank-additional-full.parquet')

# %% [markdown]
# # Available columns
#
# Firstly a short description of all the input features, taken from the provided data information.
#
# ## bank client data:
#
# 1. age (numeric)
# 2. job : type of job (categorical: "admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown")
# 3. marital : marital status (categorical: "divorced","married","single","unknown"; note: "divorced" means divorced or widowed)
# 4. education (categorical: "basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown")
# 5. default: has credit in default? (categorical: "no","yes","unknown")
# 6. housing: has housing loan? (categorical: "no","yes","unknown")
# 7. loan: has personal loan? (categorical: "no","yes","unknown")
#
# ## related with the last contact of the current campaign:
# 8. contact: contact communication type (categorical: "cellular","telephone")·
# 9. month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
# 10. day_of_week: last contact day of the week (categorical: "mon","tue","wed","thu","fri")
#
# ## other attributes:
# 11. campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
# 12. pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
# 13. previous: number of contacts performed before this campaign and for this client (numeric)
# 14. poutcome: outcome of the previous marketing campaign (categorical: "failure","nonexistent","success")
#
# ## social and economic context attributes
# 15. emp.var.rate: employment variation rate - quarterly indicator (numeric)
# 16. cons.price.idx: consumer price index - monthly indicator (numeric)
# 17. cons.conf.idx: consumer confidence index - monthly indicator (numeric)
# 18. euribor3m: euribor 3 month rate - daily indicator (numeric)
# 19. nr.employed: number of employees - quarterly indicator (numeric)

# %% [markdown]
# # EDA
#
# The first step is to get an overview of the data. For that I use seaborns pairplot, which creates scatter plots
# for each pair of columns.

# %%
sns.pairplot(data, hue='y');

# %% [markdown]
# Now I go through every feature and check for:
# * any outliers
# * if some categorial variables have too many classes
# * dependence to target variable
# * if any domain transformation would help to make the distribution more 'normal'
# * anything which looks interesting

# %% [markdown]
# ## Age
#
# * average age is similar for positive and negative case
# * but histogram shows that clients above 60, have a higher chance of accepting the offer

# %%
# mean and median age vs. target
data.groupby('y')['age'].agg(['mean', 'median', 'std'])

# %%
data.groupby('y')['age'].hist(bins=50, alpha=0.5, density=True)

# %%
# prob. for customers above 60 is 4 times higher
data[data['age'] > 60]['y'].mean()/data['y'].mean()

# %% [markdown]
# ## job
#
# * students and retired have highest prob.
# * all categories have enough samples to be used for training

# %%
px.bar(
    data['job'].value_counts(normalize=True),
    labels={'value': 'count', 'index': 'job'}
)

# %%
px.bar(
    data.groupby(['job'])['y'].mean(),
    labels={'value': 'prob'},
).show()

# %% [markdown]
# ## marital
#
# * very few 'unknown' cases: maybe replace with married (mode) or single (similar prob.);

# %%
data['marital'].value_counts()

# %%
px.bar(
    data.groupby(['marital'])['y'].mean(),
    labels={'value': 'prob'},
).show()

# %% [markdown]
# ## education
#
# * almost no illiterate cases; high prob. is maybe due to low statistics; maybe set to unknown

# %%
data['education'].value_counts()

# %%
px.bar(
    data.groupby(['education'])['y'].mean(),
    labels={'value': 'prob'},
).show()

# %% [markdown]
# ## default
#
# * only three 'yes' cases, too few to be useful for the model; drop column or set 'yes' to 'unknown'

# %%
data['default'].value_counts()

# %%
px.bar(
    data.groupby(['default'])['y'].mean(),
    labels={'value': 'prob'},
).show()

# %% [markdown]
# ## loan
#
# * enough samples for each class
# * very similar prob. in all classes

# %%
data['loan'].value_counts()

# %%
px.bar(
    data.groupby(['loan'])['y'].mean(),
    labels={'value': 'prob'},
).show()

# %% [markdown]
# ## contact
#
# * quite a huge difference in prob. between 'cellular' and 'telephone'

# %%
data['contact'].value_counts()

# %%
px.bar(
    data.groupby(['contact'])['y'].mean(),
    labels={'value': 'prob'},
).show()

# %% [markdown]
# ## month
#
# * very different number of attempts, peak in may,
# * especially the month with a lot of attemps have a very low prob; maybe the amount influences the likelohod of success
#   which means the feature might not be useful for prediction, and result on test set may be flawed

# %%
px.bar(
    data['month'].value_counts().reindex(['mar', 'apr', 'may', 'jun', 'jul', 'sep', 'oct', 'nov', 'dec']),
    labels={'value': 'count', 'index': ''}
)

# %%
px.bar(
    data.groupby(['month'])['y'].mean().reindex(['mar', 'apr', 'may', 'jun', 'jul', 'sep', 'oct', 'nov', 'dec']),
    labels={'value': 'prob'},
).show()

# %%
data['month'].value_counts().to_frame('COUNT').join(
    data.groupby(['month'])['y'].mean().to_frame('PROB')
).reindex(['mar', 'apr', 'may', 'jun', 'jul', 'sep', 'oct', 'nov', 'dec'])

# %%
# prob. versus number of attempts per month
# very high dependence
data['month'].value_counts().to_frame('COUNT').join(
    data.groupby(['month'])['y'].mean().to_frame('PROB')
)

# %%
data['month'].value_counts().to_frame('COUNT').join(
    data.groupby(['month'])['y'].mean().to_frame('PROB')
).corr()

# %% [markdown]
# ## day_of_week
#
# * equally distributed over 5 work days

# %%
data['day_of_week'].value_counts()

# %%
px.bar(
    data.groupby(['day_of_week'])['y'].mean(),
    labels={'value': 'prob'},
).show()

# %% [markdown]
# ## campaign
#
# * no outliers
# * could be used as is

# %%
data.groupby('y')['campaign'].agg(['mean', 'median', 'std'])

# %%
data.groupby('y')['campaign'].hist(bins=30, alpha=0.5, density=True, log=True)

# %% [markdown]
# ## pdays
#
# * 999 -> no previous contacts
# * may try to set 999 to -1

# %%
data.groupby('y')['pdays'].hist(bins=np.arange(0, 50), alpha=0.5, density=True);

# %% [markdown]
# ## previous
#
# * no outliers

# %%
data.groupby('y')['previous'].hist(bins=np.arange(0, 50), alpha=0.5, density=True, log=True);

# %% [markdown]
# ## poutcome
#
# * if previous attempt was a success, new attempt has a high success prob.

# %%
data['poutcome'].value_counts()

# %%
px.bar(
    data.groupby(['poutcome'])['y'].mean(),
    labels={'value': 'prob'},
).show()

# %% [markdown]
# ## social and economic context attributes
#
# All following context attributes are measured either quaterly or monthly. This means they are highly correlated to
# the month, from which we already know that it might be flawed. Month with less attempts have a much higher
# success rate than month with more attempts; this may translate to these attributes.

# %% [markdown]
# ## emp.var.rate
#
# * low cardinatity: could be used as categorial
# * lower value -> higher prob., but fluctuates quite a log

# %%
data['emp.var.rate'].value_counts().sort_index()

# %%
data.groupby('emp.var.rate')['y'].mean()

# %%
data.groupby('y')['emp.var.rate'].hist(bins=10, alpha=0.5, density=True);

# %% [markdown]
# ## cons.price.idx
#
# * no 

# %%
data.groupby('cons.price.idx')['y'].mean().to_frame().join(
    data.groupby('cons.price.idx').size().to_frame('Count')
)

# %%
px.scatter(data.groupby('cons.price.idx')['y'].mean())

# %%
data.groupby('y')['cons.price.idx'].hist(bins=10, alpha=0.5, density=True);

# %% [markdown]
# ## cons.conf.idx

# %%
data['cons.conf.idx'].value_counts().sort_index()

# %%
data.groupby('y')['cons.conf.idx'].hist(bins=10, alpha=0.5, density=True);

# %% [markdown]
# ## euribor3m
#
# * low cardinality: maybe use as categories

# %%
data.groupby('y')['euribor3m'].hist(bins=10, alpha=0.5, density=True);

# %% [markdown]
# ## nr.employed

# %%
data.groupby('y')['nr.employed'].hist(bins=10, alpha=0.5, density=True);

# %%
