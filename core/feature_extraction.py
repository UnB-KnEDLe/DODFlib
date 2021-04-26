import os
import re
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Iterable, List, Tuple, Dict, Callable, Union

def crf_sentence_featurizer(tokens: Iterable[str]):
    """Generates features from sequence of tokens.

    Args:
        tokens (Iterable[str]): [description]

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


    def __init__(self, sent_featurizer: Callable = None, lazy: bool = True):
        self.text_matrix  = []
        self.sent_featurizer = sent_featurizer or crf_sentence_featurizer
        self.lazy = lazy
    

    def fit(self, X = None, y=None, **fit_params):
        return self


    def transform(self, X: Iterable[Iterable[str]]]):
        if self.lazy:
            X = map(self.sent_featurizer, X)
        else:
            X = [self.sent_tokenizer(row) for row in X]
        return X
    

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

        if isinstance(X, pd.DataFrame):
            X = X.apply()
        if self.lazy:
            X = map(self.sent_featurizer, X)
        else:
            X = [self.sent_tokenizer(row) for row in X]
        return X
    

