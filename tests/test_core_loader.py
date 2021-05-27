import os
import re
import pytest
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from unittest.mock import patch

from dodflib.core import loader as L
from pathlib import Path

BASE_PATH = Path('support/test_core_loader')
@pytest.fixture(scope='module')
def test_path():
    return 
def test_path2str_1():
    assert 'teste.pdf' == L.path2str('teste.pdf')


def test_path2str_2():
    p = Path('dir/teste.pdf')
    assert p.as_posix() == 'dir/teste.pdf'


@pytest.fixture(scope='module')
def xml_files():
    return {'support/t1.xml', 'support/t2.xml', 'support/xml/t1.xml'}


@pytest.fixture(scope='module')
def xml_files_path_1():
    return {
        BASE_PATH/'t1.xml', 
        BASE_PATH/'t2.xml', 
        BASE_PATH/'xml/t1*.xml'}

@pytest.fixture(scope='module')
def xml_files_path_2():
    return {
        BASE_PATH/'t1.xml', 
        BASE_PATH/'t2.xml',}


def test_gen_all_files_1(xml_files_path_1):
    files = L.gen_all_files('support')
    assert set( files ) == xml_files_path_1


def test_gen_all_files_2(xml_files_path_2):
    files = list(L.gen_all_files(BASE_PATH, r'^([a-z0-9])+\.xml'))
    assert set( files ) == xml_files_path_2


def test_loaderxml_single():
    err = []
    lt2 = L.LoaderXML(BASE_PATH /'t2.xml')
    for act in lt2.act_names:
        pq = pd.read_parquet(BASE_PATH / f't2_{act}.parquet')
        c1, c2 = sorted(pq.columns), sorted(lt2[act].columns)
        if not np.all(pq[c1].fillna(0) == lt2[act][c2].fillna(0)).astype(bool):
            err.append((act, pq))
    assert not err


def test_robust_concat_1():
    cols = ['nada', 'a', 'ver']
    empty_df = L.LoaderXML._robust_concat([], columns=cols)
    assert empty_df.empty and np.all(empty_df.columns == pd.Index(cols))

