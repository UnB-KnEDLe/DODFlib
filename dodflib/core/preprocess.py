import os
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Iterable as Iter
from sklearn.base import BaseEstimator, TransformerMixin

from dodflib.core import utils
from dodflib.core import feature_extraction


class DropPattern(TransformerMixin, BaseEstimator):
    def __init__(self, pat: str):
        self.pat = pat
    def fit(self, X, y=None, **kwargs):
        return self
    def transform(self, X: pd.DataFrame, **kwargs):
        utils.type_checking(X, 'X', (pd.DataFrame,) )
        return X.drop(X.filter(regex=self.pat).columns, axis=1)


class SelectColumn(TransformerMixin, BaseEstimator):
    def __init__(self, c):
        self.c = c
    def fit(self, X, y=None, **kwargs):
        return self
    def transform(self, X: pd.DataFrame, **kwargs):
        utils.type_checking(X, 'X', (pd.DataFrame,) )
        return X[self.c]


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


class IOBifyer(TransformerMixin, BaseEstimator):


    @staticmethod
    def find_entity(row, token, ignore_idx=0,
        tokenizer=feature_extraction.DEFAULT_TOKENIZER):
        # TODO: aceitar opção de offset, para não ter tennhum tipo de problema
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
    def generate_IOB_labels(row, idx, tokenizer, dbg={}):
        """[summary]

        Args:
            row ([pd.Series]): [pandas series having act text and entities text]
            idx ([int]): [index such that `row[idx]` has the whole act]
            tokenizer ([Callable]): [function to use to tokenize `row[idx]`]
            dbg (dict, optional): [dictionay for debug purposes]. Defaults to {}.

        Returns:
            [Iter[Iter[str]]]: [matrix of IOB labels]
        """
        labels = []
        entity_started = False
        text = row.iloc[idx]
        for token in tokenizer(text):                         # Itera sobre cada token da anotação do ato.
            if not entity_started:                               # Caso uma entidade ainda n tenha sido identificada nos tokens.
                entity = IOBifyer.find_entity(row, token, idx)                 # Busca o token atual no primeiro token de todos os campos do df.
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
        if labels[0] != 'O':
            dbg['l'] = dbg.get('l', []) + [(row, idx)]

        return labels


    @staticmethod
    def dump_iob(tokens_mat, labels_mat, path='dump.txt',
                            sep=' X X ', sent_sep='\n',):
        """This method dumps the token matrix according its IOB labels.

        For debug purposes, a list of list of pairs (token, label) is returned.
        Args:
            tokens_mat ([Iter[Iter[str]]]): [matrix of strings corresponding to tokens]
            labels_mat ([Iter[Iter[str]]]): [matrix of strings corresponding to IOB labels]
            path (str, optional): [Path to dump text file]. Defaults to 'dump.txt'.
            sep (str, optional): [description]. Defaults to ' X X '.
            sent_sep (str, optional): [description]. Defaults to '\n'.

        Returns:
            [List[LIst[Tuple(str, str)]]]: [list of list of pairs (token, label), as dumped. For debug purposes.]
        """
        dbg_mat = []
        if isinstance(path, Path):
            path = path.as_posix()
        if '/' in path:
            os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)

        with open(path, 'w') as fp:
            for tokens_lis, labels_lis in zip(tokens_mat, labels_mat):
                dbg_mat.append([])
                for token, label in zip(tokens_lis, labels_lis):
                    dbg_mat[-1].append((token, label))
                    fp.write(f"{token}{sep}{label}\n")
                fp.write(sent_sep)
        return dbg_mat


    def __init__(self, column='act_column',
        tokenizer=feature_extraction.DEFAULT_TOKENIZER):
        self.column = column
        self.tokenizer = tokenizer
        self.dbg = {}


    def fit(self, X=None, y=None, **fit_params):
        return self


    def transform(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"`df` expected to be a pd.DataFrame. Got {type(df)}")
        if df.empty:
            print("[core.preprocess]Warning: empty DataFrame. There won't be ioblabels.")
            return pd.Series()

        idx = self.column if isinstance(self.column, int) else  \
                df.columns.get_loc(self.column)
        labels_row = []
        for index, row in df.iterrows():
            try:
                labels_row.append(
                    IOBifyer.generate_IOB_labels(
                        row, idx, self.tokenizer, self.dbg
                    )
                )
            except Exception as e:
                print("problem iobifyin row:", row)
                raise e
        return pd.Series(labels_row)


