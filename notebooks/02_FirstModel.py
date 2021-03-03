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
# # First Model
#
# The idea is to create a first model as quick as possible. This will give not the best results,
# but allows to iterate fast and deliver already a prototypic API later on.

# %%
import pandas as pd
import numpy as np
from IPython.display import Markdown
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as me
import category_encoders as ce

# %%
data = pd.read_csv('../data/bank-additional-full.csv', sep=';')

# %%
# the data information says, that duration should not be used, since it
# is only available after the marketing call
data = data.drop(columns=['duration'])

# %%
data['y'] = data['y'].map({'no': 0, 'yes': 1})

# %% [markdown]
# ## train - test  split
#
# Before starting to train the model the data is split into a train and a test set.
# There is a lot wrong with this simple appraoch here, but the model we build firstly will be thrown away anyways.
# In a later phase one has to correct this in two (and a half) ways:
# * we need a speparate set for the model validation and hyperparameter optimization
# * due to the low number of positive cases, one should use a startified split, such that
#   each set has enough positive samples
# * the total number of samples is not too high, so for validation a KFold approach would be advantageous

# %%
train, test = train_test_split(data, test_size=0.2, random_state=42)

# %% [markdown]
# As as little as possible time should go into the model building, the categorical variables are one-hot encoded.

# %%
cat_cols = train.dtypes[train.dtypes == 'object'].index

# %%
enc = ce.OneHotEncoder(return_df=True, cols=cat_cols, drop_invariant=True)

# %%
train = enc.fit_transform(train)
test = enc.transform(test)

# %%
x_train = train.drop(columns=['y'])
y_train = train['y']
x_test = test.drop(columns=['y'])
y_test = test['y']

# %%
clf = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    max_depth=10,
)

# %%
clf.fit(x_train, y_train)

# %% [markdown]
# ## Performance metrics
#
# As above, not details or exhausive work here. Just as fast as possible to something which is not total nonsense.
# Speaking of total nonsense, using the accuracy here would be total nonsense due to the class imbalance.
# However, for now the balanced accuracy is used as the next best thing. Although, not the best metric
# for this task.

# %%
pred_train = clf.predict(x_train)
pred_test = clf.predict(x_test)

# %%
Markdown(
f"""
Balanced accuracy on the training data {me.balanced_accuracy_score(y_train, pred_train):.3f}

Balanced accuracy on the test data {me.balanced_accuracy_score(y_test, pred_test):.3f}

Using a plain RandomForestClassifier where only the max_depth is restircted to 10,
there overfitting is ~~ok~~ not too bad.

Honestly, also the confusion matrix and the precision-recall curve looks quite nice.
Much better than I would have expected. Usually, these tasks are a little bit harder.
""")

# %%
me.plot_confusion_matrix(clf, x_test, y_test, normalize='true')

# %%
me.plot_precision_recall_curve(clf, x_test, y_test)

# %% [markdown]
# ## Final Training
#
# Finally, the model is trained on the full dataset and stored in a pickle file.

# %%
x = data.drop(columns=['y'])
y = data['y']

x = enc.fit_transform(x, y)
clf.fit(x, y)

# %%
import pickle
from pathlib import Path

# %%
model_dir = Path('../app/model/')
model_dir.mkdir(exist_ok=True, parents=True)

# %%
with open(model_dir / 'simple_enc.pkl', 'wb') as handle:
    pickle.dump(enc, handle)
    
with open(model_dir / 'simple_rf.pkl', 'wb') as handle:
    pickle.dump(clf, handle)
