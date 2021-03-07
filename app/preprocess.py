import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin


# TODO Split in separate transformers and find a better name
class CustomTransformer(TransformerMixin):
    """Transform bank marketing input data"""
    def __init__(self,
                 job={'replace_unknown': 'mode'},
                 marital={'replace_unknown': False},
                 education={'replace_unknown': False,
                            'replace_illiterate': 'mode'},
                 default={'replace_yes': 'unknown'}
                 ):

        self.job = job
        self.marital = marital
        self.education = education

    def fit(self, X, y, **fit_params):
        self.job_mode = X['job'].mode().values[0]
        self.job_categories = set(X['job'].unique())

        self.marital_mode = X['marital'].mode().values[0]
        self.marital_categories = set(X['marital'].unique())

        self.education_mode = X['education'].mode().values[0]
        self.education_categories = set(X['education'].unique())

        self.default_categories = set(X['default'].unique())

    def transform(self, X, **trans_params):
        X = X.copy()

        # input data should not contain any NA's
        assert X.insa().sum().sum() == 0

        # age
        assert (X['age'] >= 0).all()

        # job
        # ensure that no new categories are in X
        assert (set(X['job'].unique()) - self.job_categories) == set()

        # replace 'unknown' class
        if self.job['replace_unknown'] == 'mode':
            X['job'] = X['job'].replace({'unknown': self.job_mode})
        elif not self.job['replace_unknown']:
            # CAUTION: may be not possibel, since unknown has only 80 samples in training set
            pass
        else:
            raise NameError("Unknown config for job['replace_unknown']")

        # martial
        assert (set(X['marital'].unique()) - self.marital_categories) == set()

        if self.marital['replace_unknown'] == 'mode':
            X['marital'].replace({'unknown': self.marital_mode})
        elif not self.marital['replace_unknown']:
            pass
        else:
            raise NameError("Unknown config for martial['replace_unknown']")

        # education
        assert (set(X['education'].unique()) - self.education_categories) == set()

        if self.education['replace_unknown'] == 'mode':
            X['education'].replace({'unknown': self.education_mode})
        elif not self.education['replace_unknown']:
            pass
        else:
            raise NameError("Unknown config for martial['replace_unknown']")

        if self.education['replace_illiterate'] == 'mode':
            X['education'].replace({'illiterate': self.education_mode})
        else:
            raise NameError("Unknown config for martial['replace_illiterate']")

        # default
        assert (set(X['default'].unique()) - self.default_categories) == set()

        if self.education['replace_yes'] == 'unknown':
            X['default'].replace({'yes': 'unknown'})
        else:
            raise NameError("Unknown config for martial['replace_illiterate']")

        return X
