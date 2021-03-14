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

import unittest

# %run ../../app/preprocess.py

# %%
data = pd.read_parquet('../../data/bank-additional-full.parquet')

# %%
xtr = CustomTransformer().fit_transform(X=data, y=data['y'])


# %%
class TestCustomTransformer(unittest.TestCase):
    
    def test_run_on_full_dataset(self):
        """Just runs the CustomTransformer on full dataset"""
        xtr = CustomTransformer().fit_transform(X=data, y=data['y'])
        
        self.assertEqual(len(data), len(xtr))
        self.assertFalse(xtr.empty)
        
    def test_drop_nr_employed(self):
        xtr = CustomTransformer(nr_employed={'drop': False}).fit_transform(X=data, y=data['y'])
        self.assertTrue('nr.employed' in xtr.columns)
        
        xtr = CustomTransformer(nr_employed={'drop': True}).fit_transform(X=data, y=data['y'])
        self.assertFalse('nr.employed' in xtr.columns)
        
    def test_drop_cons_conf_idx(self):
        xtr = CustomTransformer(cons_conf_idx={'drop': False}).fit_transform(X=data, y=data['y'])
        self.assertTrue('cons.conf.idx' in xtr.columns)
        
        xtr = CustomTransformer(cons_conf_idx={'drop': True}).fit_transform(X=data, y=data['y'])
        self.assertFalse('cons.conf.idx' in xtr.columns)
        
    def test_drop_cons_price_idx(self):
        xtr = CustomTransformer(cons_price_idx={'drop': False}).fit_transform(X=data, y=data['y'])
        self.assertTrue('cons.price.idx' in xtr.columns)
        
        xtr = CustomTransformer(cons_price_idx={'drop': True}).fit_transform(X=data, y=data['y'])
        self.assertFalse('cons.price.idx' in xtr.columns)
        
    def test_drop_emp_var_rate(self):
        xtr = CustomTransformer(emp_var_rate={'drop': False}).fit_transform(X=data, y=data['y'])
        self.assertTrue('emp.var.rate' in xtr.columns)
        
        xtr = CustomTransformer(emp_var_rate={'drop': True}).fit_transform(X=data, y=data['y'])
        self.assertFalse('emp.var.rate' in xtr.columns)  
        
    def test_drop_month(self):
        xtr = CustomTransformer(month={'drop': False}).fit_transform(X=data, y=data['y'])
        self.assertTrue('month' in xtr.columns)
        
        xtr = CustomTransformer(month={'drop': True}).fit_transform(X=data, y=data['y'])
        self.assertFalse('month' in xtr.columns)  
        
    def test_replace_pdays(self):
        xtr = CustomTransformer(pdays={'replace_999': False}).fit_transform(X=data, y=data['y'])
        self.assertTrue(999 in xtr['pdays'].unique())
        
        xtr = CustomTransformer(pdays={'replace_999': '-1'}).fit_transform(X=data, y=data['y'])
        self.assertFalse(999 in xtr['pdays'].unique())
        self.assertTrue(-1 in xtr['pdays'].unique())
        
    def test_replace_unknown_load(self):
        xtr = CustomTransformer(loan={'replace_unknown': False}).fit_transform(X=data, y=data['y'])
        self.assertTrue('unknown' in xtr['loan'].unique())
        
    def test_replace_yes_default(self):
        xtr = CustomTransformer(default={'replace_yes': 'unknown'}).fit_transform(X=data, y=data['y'])
        self.assertTrue('unknown' in xtr['default'].unique())
        self.assertFalse('yes' in xtr['default'].unique())
        
    def test_replace_education(self):
        xtr = CustomTransformer(
            education={'replace_unknown': 'mode', 'replace_illiterate': 'mode'}
        ).fit_transform(X=data, y=data['y'])
        
        self.assertFalse('unknown' in xtr['education'].unique())
        self.assertFalse('illiterate' in xtr['education'].unique())
        
        xtr = CustomTransformer(
            education={'replace_unknown': False, 'replace_illiterate': 'mode'}
        ).fit_transform(X=data, y=data['y'])
        
        self.assertTrue('unknown' in xtr['education'].unique())
        self.assertFalse('illiterate' in xtr['education'].unique())
        
    def test_replace_unknown_job(self):
        xtr = CustomTransformer(job={'replace_unknown': 'mode'}).fit_transform(X=data, y=data['y'])
        self.assertFalse('unknown' in xtr['job'].unique())
        
        xtr = CustomTransformer(job={'replace_unknown': False}).fit_transform(X=data, y=data['y'])
        self.assertTrue('unknown' in xtr['job'].unique())
        
    def test_replace_unknown_martial(self):
        xtr = CustomTransformer(marital={'replace_unknown': 'mode'}).fit_transform(X=data, y=data['y'])
        self.assertFalse('unknown' in xtr['marital'].unique())
        
        xtr = CustomTransformer(marital={'replace_unknown': False}).fit_transform(X=data, y=data['y'])
        self.assertTrue('unknown' in xtr['marital'].unique())


# %%
unittest.main(argv=[''], verbosity=2, exit=False)
