import os 
import re
import pandas as pd
import numpy as np
import attr
from pathlib import Path
from typing import Iterable



def gen_all_files(root, pat=r'.*[.]py$'):

    """Generator for all file paths matching regex pattern `pat`.

    Generates `all` files matching `pat` inclus
    Yields:
        [type]: [description]
    """
    lis = []
    for root, dirs, files in os.walk(root, topdown=True):
       root = Path(root)
       for f in files:
           if re.search(pat, f):
               yield Path(root/f)


def base_assembler(seq: Iterable[pd.DataFrame]):
    """Stacks dataframes.

    Args:
        seq (Iterable[pd.DataFrame]): an iterable containing only pd.DataFrames 

    Returns:
        pd.DataFrame: a big DataFrame having all the data
    """
    return pd.concat(seq) 
            


@attr.s
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
    root_or_data = attr.ib()
    parser = attr.ib(default=None)
    assembler = attr.ib(default=None)
    pattern = attr.ib(default=None)


    def fit(self, *args, **kwargs):
        return self


    def transform(self, root_or_data=None):
        self.root_or_data = root_or_data or self.root_or_data
        if isinstance(self.root_or_data, (str, Path)):
            if os.path.isdir(self.root_or_data):
                generator = gen_all_files(self.root_or_data, self.pattern)
                assembled = assembler(map(self.parser, generator))
            elif os.path.isfile(self.root_or_data):
                assembled = assembler(map(self.parser, [Path(self.root_or_data)]))
            else:
                raise ValueError("`self.root_or_data` must be an path to file or folder.")
        elif isinstance(self.root_or_data, (pd.DataFrame, )):
            assembled = self.root_or_data.copy()
        return assembled


    def load(self, root_or_data=None):
        return self.transform(root_or_data)
class LoaderXML:
    def __init__(self):
        pass