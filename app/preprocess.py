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
                 default={'replace_yes': 'unknown'},
                 loan={'replace_unknown': False},
                 month={'drop': True},
                 pdays={'replace_999': False},
                 emp_var_rate={'drop': False},
                 cons_price_idx={'drop': False},
                 cons_conf_idx={'drop': False},
                 nr_employed={'drop': False}
                 ):

        self.job = job
        self.marital = marital
        self.education = education
        self.default = default
        self.loan = loan
        self.month = month
        self.pdays = pdays
        self.emp_var_rate = emp_var_rate
        self.cons_price_idx = cons_price_idx
        self.cons_conf_idx = cons_conf_idx
        self.nr_employed = nr_employed

    def fit(self, X, y, **fit_params):
        self.job_mode = X['job'].mode().values[0]
        self.job_categories = set(X['job'].unique())

        self.marital_mode = X['marital'].mode().values[0]
        self.marital_categories = set(X['marital'].unique())

        self.education_mode = X['education'].mode().values[0]
        self.education_categories = set(X['education'].unique())

        self.default_categories = set(X['default'].unique())

        self.loan_categories = set(X['loan'].unique())

        self.contact_categories = set(X['contact'].unique())

        self.doy_categories = set(X['day_of_week'].unique())

        self.poutcome_categories = set(X['poutcome'])

        return self

    def transform(self, X, **trans_params):
        X = X.copy()

        # input data should not contain any NA's
        assert X.isna().sum().sum() == 0

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
            X['marital'] = X['marital'].replace({'unknown': self.marital_mode})
        elif not self.marital['replace_unknown']:
            pass
        else:
            raise NameError("Unknown config for martial['replace_unknown']")

        # education
        assert (set(X['education'].unique()) - self.education_categories) == set()

        if self.education['replace_unknown'] == 'mode':
            X['education'] = X['education'].replace({'unknown': self.education_mode})
        elif not self.education['replace_unknown']:
            pass
        else:
            raise NameError("Unknown config for education['replace_unknown']")

        if self.education['replace_illiterate'] == 'mode':
            X['education'] = X['education'].replace({'illiterate': self.education_mode})
        else:
            raise NameError("Unknown config for education['replace_illiterate']")

        # default
        assert (set(X['default'].unique()) - self.default_categories) == set()

        if self.default['replace_yes'] == 'unknown':
            X['default'] = X['default'].replace({'yes': 'unknown'})
        else:
            raise NameError("Unknown config for default['replace_yes']")

        # loan
        assert (set(X['loan'].unique()) - self.loan_categories) == set()

        if not self.loan['replace_unknown']:
            pass
        else:
            raise NameError("Unknown config for loan['replace_unknown']")

        # contact
        assert (set(X['contact'].unique()) - self.contact_categories) == set()

        # month
        if self.month['drop']:
            X = X.drop(columns=['month'])
        else:
            pass

        # day_of_week
        assert (set(X['day_of_week'].unique()) - self.doy_categories) == set()

        # campaign
        assert (X['campaign'] >= 0).all()

        # pdays
        assert (X['pdays'] >= 0).all()

        if not self.pdays['replace_999']:
            pass
        elif self.pdays['replace_999'] == '-1':
            X['pdays'] = X['pdays'].replace({999: -1})
        else:
            raise NameError("Unknown config for loan['replace_unknown']")

        # previous
        assert (X['previous'] >= 0).all()

        # poutcome
        assert (set(X['poutcome'].unique()) - self.poutcome_categories) == set()

        drop_cols = []

        # cons.conf.idx
        if self.cons_conf_idx['drop']:
            drop_cols.append('cons.conf.idx')

        # emp.var.rate
        if self.emp_var_rate['drop']:
            drop_cols.append('emp.var.rate')

        # cons.price.idx
        if self.cons_price_idx['drop']:
            drop_cols.append('cons.price.idx')

        # cons.conf.idx
        if self.cons_conf_idx['drop']:
            drop_cols.append('cons.conf.idx')

        # euribor3m
        # drop not useful

        # nr.employed
        if self.nr_employed['drop']:
            drop_cols.append('nr.employed')

        X = X.drop(columns=drop_cols)

        return X
