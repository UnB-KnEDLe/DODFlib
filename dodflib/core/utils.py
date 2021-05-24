import os
import xml
import glob
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Union, Tuple, Iterable as Iter


def extract_text_and_offset(annot: ET):
    """Extracts annotation from XML node.

    Args:
        annot (xml.etree.ElementTree): XML node

    Returns:
        Tuple[str, str, str, int]:
            - type of annotation
            - text of annotation
            - name of annotator
            - offset from the beggining of the document
    """
    type_annot = annot.find('.//infon[@key="type"]').text
    text_annot = annot.find('text').text
    annotator = annot.find('.//infon[@key="annotator"]').text
    offset_annot = int(annot.find('location').attrib['offset'])
    return type_annot, annotator, text_annot, offset_annot


def xml2dictlis(root: Union[str, Path, ET.Element], prefix='Ato_'):
    """Extract XML relations to list.

    Note1: the `*_annotator` column do not add new information
    about the act, however should be useful to tasks such as
    tracking annotators progress, knowing which annotator anno-
    tates better, etc.
    Note2: the `*_offset` columns may be useful in cases having
    more than one occurence of some substring.
    Note3: all

    Args:
        root (Union[str, Path, ET.element]): path to XML file or XML node
        prefix ([str]): filter to decide which nodes will be retrieved
    Raises:
        TypeError: raised when the type of `root` is innadequate

    Returns:
        lis: list of `dict` each having the keys and values of
            an kind of act
    """
    if isinstance(root, (str, Path)):
        if os.path.isdir(root):
            raise ValueError(f"`root` must be path to file, not dir. Got: {root}")
        root = ET.parse(root)
    elif not isinstance(root, ET.Element):
        raise TypeError(
            "`root` must be instance of `str`, `pathlib.Path` or \
            xml.etree.ElementTree.Element` ]"
        )
    lis = []
    for rel in root.findall('.//relation'):
        row_act = {}
        annotator_rel = rel.find('.//infon[@key="annotator"]').text
        row_act['rel_id'] = rel.attrib['id']
        row_act['rel_annotator'] = annotator_rel

        for node in rel.findall('node'):
            ref_id = node.get('refid')
            annot = root.find(f'.//annotation[@id="{ref_id}"]')
            tyype, annotator, txt, off = extract_text_and_offset(annot)
            row_act[tyype] = txt
            row_act[f'{tyype}_offset'] = off
            row_act[f'{tyype}_annotator'] = annotator
        lis.append(row_act)

    return [
        d for d in lis if
        any( [j.startswith(j) for j in d] )
    ]


def type_checking(obj, name, classes):
    if not isinstance(obj, classes):
        raise TypeError(
            f"`{name}` must be of type: {', '.join(map(str,classes))}"
        )
    return True


def pad_dataframe(df: pd.DataFrame, to_pad: Iter[str], fill_va=np.nan):
    """ 'Pad' dataframe by adding columns of `fill_va`.

    Given a pandas DataFrame, a list of 'paddable' columns,
    each paddable column not present in `df`` is added it
    having only `fill_va` values.
    Args:
        df (pd.DataFrame): pandas DataFrame to be padded
        to_pad (Iter[str]): iterable containing columns to be padded
        fill_va: an arbitrary value to be set on padded columns
    """
    type_checking(df, 'df', (pd.DataFrame,))
    type_checking(to_pad, 'to_pad', (Iter, ))
    for idx, it in enumerate(to_pad):
        type_checking(it, f'to_pad[{idx}]', (str, ))


    df_cols = set(df.columns)
    pad_cols = set(to_pad)
    df[list(pad_cols - df_cols)] = fill_va
    return df

