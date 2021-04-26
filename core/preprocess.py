import os
import re
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class Tokenizer(TransformerMixin, BaseEstimator):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    
    def __call__(self, X, **kw_params):
        return self.tokenizer(X, **kw_params)


    def fit(self, X, y=None, **fit_params):
        return self
    

    def transform(self, X, **kw_params):
        return self.tokenizer(X, **kw_params)


