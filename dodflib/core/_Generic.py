import os 
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Iterable


def drop_cols_thresh(data: pd.DataFrame, thresh = .7):
    """Drops dataframe columns with more than `thresh` missing values.

    Args:
        data (pd.DataFrame): the data to be cleansed
        thresh (float): 
            if there are more rows than this percentage of missing values,
            drop the columns
    Returns:
        pd.DataFrame: the cleansed dataframe

    """
    nan_by_rows = data.isna().sum() / data.shape[0]
    columns = [c for c in data.columns if nan_by_rows[c] > thresh]
    return data.filter(items=columns)
            


class Loader:
    """Loads data.
    
    Applies `assembler` to every file parsed by `parser`, assembling the whole data.
    Args:
        root_or_data (Union[str, Path, pd.DataFrame]):
            if Union[str, Path]:    path to root having the `input` data files,
            if pd.DataFrame: DataFrame containing the input data
        file_parser (Callable): 
            a callable that is going to be called for each data file
        pattern (Union[str, re.Pattern]):
            pattern to be matched against each file under `root_or_data`
        assembler (Callable):
            callable responsible for processing all files found and
            to glue it all together 

    """
    def __init__(self, root_or_data, parser, assembler, pattern):
        self.root_or_data = root_or_data or self.root_or_data
        self.parser = parser
        self.assembler = assembler
        self.pattern = pattern


    def fit(self, *args, **kwargs):
        return self


    def transform(self, root_or_data=None):
        
        if isinstance(self.root_or_data, (str, Path)):
            if os.path.isdir(self.root_or_data):
                generator = gen_all_files(self.root_or_data, self.pattern)
                assembled = assembler(map(self.parser, generator))
            elif os.path.isfile(self.root_or_data):
                assembled = assembler(map(self.parser, [Path()]))
            else:
                raise ValueError("`self.root_or_data` must be an path to file or folder.")
        elif isinstance(self.root_or_data, (pd.DataFrame, )):
            assembled = self.root_or_data.copy()
        return assembled


    def load(self, root_or_data=None):
        return self.transform(root_or_data)


