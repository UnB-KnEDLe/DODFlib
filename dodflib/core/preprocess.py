import os
import re
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import nltk

DEFAULT_TOKENIZER = nltk.RegexpTokenizer(r"\w+").tokenize

class Tokenizer(TransformerMixin, BaseEstimator):
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
            input()
            X = pd.Series(X)
        return X.map(self.tokenizer)


class KeepOnlyOneAct(TransformerMixin, BaseEstimator):
    
    def __init__(self, act_name, pattern: str = r'^Ato_'):
        self.act_name = act_name
        self.pattern = pattern


    def fit(self, X, y=None, **fit_params):
        return self
    

    def transform(self, X: pd.DataFrame, **kw_params):

        if not isinstance(X, (pd.DataFrame, )):
            raise TypeError("`X` must a pandas DataFrame")
        acts_cols = set(X.filter(regex=self.pattern).columns)
        if self.act_name in acts_cols:
            acts_cols.remove(self.act_name)

        return X.drop(acts_cols, axis=1)




class IOB_Transformer:


    @staticmethod
    def find_entity(row, token, ignore_idx=0, 
        tokenizer=DEFAULT_TOKENIZER):
        """Searches for named entities on columns, except by ignore_idx-columns.
        
        ignore_idx: int indicating which column has
                    the TEXT where the named entity were extracted from
        """
        for idx, column in enumerate(row.keys()):
            if idx == ignore_idx:
                continue
            if isinstance(row[column], str) and \
                token == tokenizer(row[column])[0]:
                return column
        
        return None



    @staticmethod
    def generate_IOB_labels(row, idx, tokenizer):
        """Generates IOB-labeling for whole text and entities.

        Assumes `row` has the whole text on the first collumn
        and the remaining contain entities.
        """
        labels = []
        entity_started = False
        text = row.iloc[idx]
        for token in tokenizer(text):                         # Itera sobre cada token da anotação do ato.
            if not entity_started:                               # Caso uma entidade ainda n tenha sido identificada nos tokens.
                entity = IOB_Transformer.find_entity(row, token, idx)                 # Busca o token atual no primeiro token de todos os campos do df.
                if entity is not None:                           # Se foi encontrado o token no inicio de alguma entidade ele inicia a comparação token a token com a entidade.
                    entity_started = True
                    token_index = 1
                    labels.append('B-' + entity)
                else:
                    labels.append('O')
            else:     # Caso uma entidade já tenha sido identificada
                if token_index < len(tokenizer(row[entity])) and \
                    token == tokenizer(row[entity])[token_index]:
                    # Checa se o próximo token pertence à entidade
                    # e se o tamanho da entidade chegou ao fim.
                    labels.append('I-' + entity)
                    # Se a entidade ainda possui tokens e a comparação foi bem 
                    # sucedida adicione o label I.
                    token_index += 1
                    if token_index >= len(tokenizer(row[entity])):
                        entity_started = False
                else:
                    # Se o token n for igual ou a entidade chegou ao fim.
                    entity_started = False
                    labels.append('O')
                    
        return labels



    def __init__(self, column='act_column', tokenizer=DEFAULT_TOKENIZER):
        self.column = column
        self.tokenizer = tokenizer


    def fit(self, X=None, y=None, **fit_params):
        return self


    def transform(self, df):
        idx = self.column if isinstance(self.column, int) else  \
                df.columns.get_loc(self.column)
        labels_row = []
        for index, row in df.iterrows():
            try:
                labels_row.append(
                    ' '.join(
                        IOB_Transformer.generate_IOB_labels(
                            row, idx, self.tokenizer
                        )
                    )
                )
            except Exception as e:
                print(row)
                raise e

        return pd.Series(labels_row).str.split()
