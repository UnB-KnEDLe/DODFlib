import os
import re
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Tuple, Dict, Callable, Union
from typing import Iterable as Iter

def crf_sentence_featurizer(tokens: Iter[str]):
    """Generates features from sequence of tokens.

    Args:
        tokens (Iter[str]): [description]

    Returns:
        List[Dict]: list of dicts containing the feature of each token 
    """
    return [{
            'word': token.lower(),
            'capital_letter': token[0].isupper(),
            'all_capital': token.isupper(),
            'isdigit': token.isdigit(),
            'word_before': token.lower() if j==0 else tokens[j-1].lower(),
            'word_after:': token.lower() if j+1>=len(tokens) else tokens[j+1].lower(),
            'BOS': j==0,
            'EOS': j==len(tokens)-1
        }
        for j, token in enumerate(tokens) if token
    ]
    


class TransfomerCRF(BaseEstimator, TransformerMixin):


    def __init__(self, sent_featurizer: Callable = None,):
        self.text_matrix  = []
        # self.tokenizer = tokenizer
        self.sent_featurizer = sent_featurizer or crf_sentence_featurizer

    def fit(self, X = None, y=None, **fit_params):
        return self


    def transform(self, X: Iter[Iter[str]] ):
        if not isinstance(X, pd.Series):
            X = pd.Series(X)
        X = X.map(self.sent_featurizer)
        # X = X.map(self.tokenizer).map(self.sent_featurizer)
        return X.values
    

class TransformerIOB(BaseEstimator, TransformerMixin):


    def __init__(self, sent_featurizer: Callable = None, lazy: bool = True):
        self.sent_featurizer = sent_featurizer or crf_sentence_featurizer
        self.lazy = lazy
        self._cache = None

    def fit(self, X=None, y=None, **fit_params):
        if not isinstance(X, None):
            self._cache = X
        return self


    def transform(self, X=None):
        return X

