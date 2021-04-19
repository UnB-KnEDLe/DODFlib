import os 
import re
import pandas as pd
import numpy as np
import attr
from pathlib import Path
from typing import Dict, Iterable, List, Union 

from knedle_nlp.core import utils
from knedle_nlp.core.utils import _type_checking

Iter = Iterable
Path_T = Union[str, Path]
Pattern_T = Union[str, re.Pattern]


ITER = Iterable
PATH_T = Path_T.__dict__['__args__']
PATTERN_T = Pattern_T.__dict__['__args__']


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
    """Class aiming to facilitate extraction of XML annotated data.

    """

    @classmethod
    def load(cls, path: Union[Path_T, Iterable[Path_T]]):
        """Main class method to extract annotations from XML files.

        Args:
            path (Union[Path_T, Iterable[Path_T]]): path to one or more XML files

        Returns:
            Dict[Union[str, Path], List[Dict]]: dict mapping filename -> list of dict having 
                annotations attributes and values
        """
        if isinstance(path, PATH_T):
            return {path: utils.xml2dictlis(path)}
        elif isinstance(path, Iterable):
            if all([isinstance(i, PATH_T) for i in path]):
                return {p: utils.xml2dictlis(p) for p in path}
        else:
            raise TypeError(
                "`path` must be of type str, pathlib.Path or iterable of them"
            )

    @classmethod
    def discovery_and_load(cls, root: Path_T, pat: Pattern_T):
        """Convenience method for applying `load` on dynamically discovered files.

        Args:
            root (Path_T): where to start searching for files
            pat (Pattern_T): the pattern each file must match to be loaded

        Raises:
            TypeError: [description]

        Returns:
            [type]: [description]
        """
        _type_checking(root, 'root', PATH_T)
        _type_checking(pat, 'pat', PATTERN_T)

        return {p: utils.xml2dictlis(p) for p in gen_all_files(root, pat)}


