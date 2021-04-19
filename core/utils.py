import os
import xml
import xml.etree.ElementTree as ET
import pandas as pd
import glob
from pathlib import Path
from typing import Union, Tuple


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


def xml2dictlis(root: Union[str, Path, ET.Element]):
    """Extract XML relations to list.

    Note1: the `*_annotator` column do not add new information
    about the act, however should be useful to tasks such as 
    tracking annotators progress, knowing which annotator anno-
    tates better, etc.
    Note2: the `*_offset` columns may be useful in cases having
    more than one occurence of some string.

    Args:
        root (Union[str, Path, ET.element]): path to XML file or XML node

    Raises:
        TypeError: raised when the type of `root` is innadequate

    Returns:
        lis: list of `dict` each having the keys and values of 
            an kind of act
    """
    if isinstance(root, (str, Path)):
        root = ET.parse(root)
    elif not isinstance(root, ET.Element):
        raise TypeError(
            "`root` must be instance of `str`, `pathlib.Path` or \
            xml.etree.ElementTree.Element` ]"
        )
    lis = []
    for rel in root.findall('.//relation'):
        row_act = {}
        type_rel = rel.find('.//infon[@key="type"]').text
        annotator_rel = rel.find('.//infon[@key="annotator"]').text
        row_act['rel_id'] = rel.attrib['id']
        row_act['rel_annotator'] = annotator_rel

        for node in rel.findall('node'):
            ref_id = node.get('refid')
            annot = root.find(f'.//annotation[@id="{ref_id}"]')
            tpe, annotator, txt, off = extract_text_and_offset(annot)
            row_act[tpe] = txt
            row_act[f'{tpe}_offset'] = off
            row_act[f'{tpe}_annotator'] = annotator
        lis.append(row_act)

        del row_act
    return lis


def get_all_files(root, pat=r'.*[.]xml$'):
    lis = []
    for root, dirs, files in os.walk(root, topdown=True):
       root = Path(root)
       lis.extend( (root/f for f in files if re.search(pat, f) ) )
    return lis



def _type_checking(obj, name, classes):
    if not isinstance(obj, classes):
        raise TypeError(
            f"`{name}` must be of type: {', '.join(map(str,classes))}"
        )
    return True    

