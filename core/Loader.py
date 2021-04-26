import os 
import re
import pandas as pd
import numpy as np
import attr
from pathlib import Path
from typing import Dict, Iterable, List, Union 

from dodflib.core import utils
from dodflib.core.utils import _type_checking, pad_dataframe
import itertools

Iter = Iterable
Path_T = Union[str, Path]
Pattern_T = Union[str, re.Pattern]


ITER = Iterable
PATH_T = Path_T.__dict__['__args__']
PATTERN_T = Pattern_T.__dict__['__args__']

NonteType = type(None)

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
    all_columns = set( itertools.chain(*(cols for cols in atos_dict.values())))


    @property
    def act_names(self):
        return list(atos_dict.keys())


    def _robust_concat(self, df_lis, columns=[]):
        """Returns a pandas DataFrame of all `df_lis` dataframes.

        Args:
            df_lis ([type]): [description]
            columns ([type]): [description]

        Returns:
            [type]: [description]
        """
        dfinal = None
        try:
            dfinal = pd.concat(df_lis)
            dfinal = pad_dataframe(dfinal, to_pad=columns)
        except Exception as e:
            dfinal = pd.DataFrame([], columns=columns)
        return dfinal


    def __init__(self, path: Union[Path_T, Iterable[Path_T]]):
        """
        Args:
            path (Union[Path_T, Iterable[Path_T]]): location(s) of `xml` file(s)
        """
        self.xml_dicts = LoaderXML.load(path)
        self._by_acts = self._build_data_by_act_name()
        # self._by_path = 

    
    def _build_data_by_act_name(self):
        d = {}
        aux_cols = ['rel_id', 'rel_annotator']
        for (act_name, act_cols) in self.atos_dict.items():
            cols = aux_cols + act_cols
            df_lis = []
            for (path, lis) in self.xml_dicts.items():
                has_act_name = [d for d in lis if not pd.isna(d.get(act_name))]
                df_act_name = self._robust_concat([
                    pd.DataFrame( [ d.values() ], columns=d.keys() )
                    for d in has_act_name                
                ], cols)
                df_act_name['path'] = path
                df_act_name.set_index(['rel_id', 'path'], drop=True, inplace=True,)
                df_lis.append(df_act_name)
                
            d[act_name] = pd.concat(df_lis)
        return d



    def get(self, key: str = None):
        if key.startswith('path:'):
            return self.get_by_path(key[5:])
        else:    
            return self.get_by_act(key)


    def __getitem__(self, key):
        return self.get(key)


    def get_by_act(self, act_name: str):
        return self._by_acts[act_name]


    def get_by_path(self, path):
        lis = []
        for d in self.xml_dicts[path]:
            df = pd.DataFrame([d.values()], columns=d.keys())
            lis.append(df)
        return self._robust_concat(lis, self.all_columns).set_index(['rel_id'])


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

