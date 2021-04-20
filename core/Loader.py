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
    atos_dict = {
        'Ato_Retificacao_Comissionado':[
            'Ato_Retificacao_Comissionado',
            'data_documento',
            'data_dodf',
            'informacao_corrigida',
            'informacao_errada',
            'lotacao',
            'nome',
            'numero_dodf',
            'pagina_dodf',
            'tipo_ato',
            'tipo_documento',
            'tipo_edicao',
        ],
        'Ato_Tornado_Sem_Efeito_Exo_Nom': [
            'Ato_Tornado_Sem_Efeito_Exo_Nom',
            'cargo_comissionado',
            'cargo_efetivo',
            'data_documento',
            'data_dodf',
            'hierarquia_lotacao',
            'matricula',
            'matricula_SIAPE',
            'nome',
            'numero_dodf',
            'orgao',
            'pagina_dodf',
            'simbolo',
            'tipo_documento',
            'tipo_edicao',
        ],
        'Ato_Exoneracao_Comissionado': [
            'Ato_Exoneracao_Comissionado',
            'a_pedido_ou_nao',
            'cargo_comissionado',
            'cargo_efetivo',
            'hierarquia_lotacao',
            'matricula',
            'matricula_SIAPE',
            'motivo',
            'nome',
            'orgao',
            'simbolo',
            'vigencia',
        ],
        'Ato_Nomeacao_Comissionado': [
            'Ato_Nomeacao_Comissionado',
            'cargo_comissionado',
            'cargo_efetivo',
            'hierarquia_lotacao',
            'matricula',
            'matricula_SIAPE',
            'nome',
            'orgao',
            'simbolo',
        ],
        'Ato_Nomeacao_Efetivo': [
            'Ato_Nomeacao_Efetivo',
            'candidato',
            'candidato_PNE',
            'cargo',
            'carreira',
            'data_dodf_edital_normativo',
            'data_dodf_resultado_final',
            'data_edital_normativo',
            'data_edital_resultado_final',
            'edital_normativo',
            'edital_resultado_final',
            'especialidade',
            'numero_dodf_edital_normativo',
            'numero_dodf_resultado_final',
            'orgao',
            'processo_SEI',
        ],
        'Ato_Abono_Permanencia': [
            'Ato_Abono_Permanencia',
            'cargo_efetivo',
            'classe',
            'fundamento_legal',
            'matricula',
            'matricula_SIAPE',
            'nome',
            'orgao',
            'padrao',
            'processo_SEI',
            'quadro',
            'vigencia',
        ],
        'Ato_Substituicao': [
            'Ato_Substituicao',
            'cargo_objeto_substituicao',
            'cargo_substituto',
            'data_final',
            'data_inicial',
            'hierarquia_lotacao',
            'matricula_SIAPE',
            'matricula_substituido',
            'matricula_substituto',
            'motivo',
            'nome_substituido',
            'nome_substituto',
            'orgao',
            'simbolo_objeto_substituicao',
            'simbolo_substituto',
        ],
        'Ato_Tornado_Sem_Efeito_Apo': [
            'Ato_Tornado_Sem_Efeito_Apo',
            'cargo_efetivo',
            'classe',
            'data_documento',
            'data_dodf',
            'fundamento_legal',
            'matricula',
            'matricula_SIAPE',
            'nome',
            'numero_documento',
            'numero_dodf',
            'orgao',
            'padrao',
            'pagina_dodf',
            'processo_SEI',
            'quadro',
            'tipo_documento',
        ],
        'Ato_Retificacao_Efetivo': [
            'Ato_Retificacao_Efetivo',
            'cargo_efetivo',
            'classe',
            'data_documento',
            'data_dodf',
            'informacao_corrigida',
            'informacao_errada',
            'lotacao',
            'matricula',
            'matricula_SIAPE',
            'nome',
            'numero_documento',
            'numero_dodf',
            'padrao',
            'pagina_dodf',
            'tipo_ato',
            'tipo_documento',
            'tipo_edicao',
        ],
        'Ato_Cessao': [
            'Ato_Cessao',
            'cargo_efetivo',
            'cargo_orgao_cessionario',
            'classe',
            'fundamento_legal',
            'hierarquia_lotacao',
            'matricula',
            'matricula_SIAPE',
            'nome',
            'onus',
            'orgao_cedente',
            'orgao_cessionario',
            'padrao',
            'processo_SEI',
            'simbolo',
            'vigencia',
        ],
        'Ato_Reversao': [
            'Ato_Reversao',
            'cargo_efetivo',
            'classe',
            'fundamento_legal',
            'matricula',
            'matriucla_SIAPE',
            'motivo',
            'nome',
            'orgao',
            'padrao',
            'processo_SEI',
            'quadro',
            'vigencia',
        ],
        'Ato_Exoneracao_Efetivo': [
            'Ato_Exoneracao_Efetivo',
            'cargo_efetivo',
            'carreira',
            'classe',
            'fundamento_legal',
            'matricula',
            'matricula_SIAPE',
            'motivo',
            'nome',
            'orgao',
            'padrao',
            'processo_SEI',
            'quadro',
            'vigencia',
        ]
    }

    def __init__(self, path: Union[Path_T, Iterable[Path_T]]):
        """
        Args:
            path (Union[Path_T, Iterable[Path_T]]): location(s) of `xml` file(s)
        """
        self.xml_dicts = LoaderXML.load(path)


    def __getitem__(self, act_name: str):
        return self.get(act_name)


    def get(self, act_name: str):
        if act_name not in self.atos_dict:
            raise ValueError(f"`act_name` must be any of {', '.join(self.atos_dict)}")
        wished_columns = ['rel_id', 'rel_annotator'] + self.atos_dict.get(act_name, [])
        return {
            path: pd.concat(
                    [pd.DataFrame( [d.values()], columns=d.keys())
                        for d in lis if act_name in d]
                )
                for (path, lis) in self.xml_dicts.items()
                or pd.DataFrame([], columns=wished_columns)            
        }


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
        elif isinstance(path, Iterable) and all([isinstance(i, PATH_T) for i in path]):
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


    

# class LoaderByAct:
#     """Class to load xml files by act.
#     """


#     def __init__(self, path: Union[Path_T, Iterable[Path_T]]):
#         """
#         Args:
#             path (Union[Path_T, Iterable[Path_T]]): location(s) of `xml` file(s)
#         """
#         self.xml_dicts = LoaderXML.load(path)
    

#     def load(self, act_name: str):
#         """Utility method to get only specific `act_name` data.

#         Args:
#             act_name (str): name of act to be extracted

#         Returns:
#             [Dict[str, pd.DataFrame]]: dictionaty from path to pandas DataFrame
#         """
#         df_dict = {}
#         wished_columns = ['rel_id', 'rel_annotator'] + LoaderByAct.atos_dict[act_name]

#         for path, lis in self.xml_dicts.items():
#             # Select acts of type `act_name`
#             act_lis = [d for d in lis if act_name in d]
#             df_dict[path] = pd.DataFrame(act_lis)
#         return df_dict


