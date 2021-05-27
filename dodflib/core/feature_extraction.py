import os
import re
import nltk
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Union
from typing import Iterable as Iter
from typing import Callable
from sklearn.base import BaseEstimator, TransformerMixin


DEFAULT_TOKENIZER = nltk.RegexpTokenizer(r"\w+").tokenize
class Tokenizer(TransformerMixin, BaseEstimator):
    """ Class to apply tokenizer to pandas DataFrame.
    """
    def __init__(self, tokenizer=DEFAULT_TOKENIZER):
        self.tokenizer = tokenizer


    def __call__(self, X, **kw_params):
        return self.tokenizer(X, **kw_params)


    def fit(self, X, y=None, **fit_params):
        return self


    def transform(self, X, **kw_params):
        if not isinstance(X, pd.Series):
            print("[preprocess.Tokenizer.transform] TYPE:", type(X))
            print('X:::: ', X)
            X = pd.Series(X)
        return X.map(self)


class CRFFeaturizer(BaseEstimator, TransformerMixin):

    @classmethod
    def crf_sentence_featurizer(cls, tokens: Iter[str]):
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



    def __init__(self, sent_featurizer: Callable = None,):
        self.text_matrix  = []
        self.sent_featurizer = sent_featurizer or self.crf_sentence_featurizer


    def fit(self, X = None, y=None, **fit_params):
        return self


    def transform(self, X: Iter[Iter[str]] ):
        if not isinstance(X, pd.Series):
            X = pd.Series(X)
        X = X.map(self.sent_featurizer)
        return X.values

