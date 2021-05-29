import torch
import operator
import unicodedata
import numpy as np
from bisect import bisect

from gensim.models import KeyedVectors
from torch.nn.utils.rnn import pad_sequence

from pathlib import Path
import pandas as pd
from collections import Counter


def find_entities(tag):
    entities = []
    prev_tag = 1
    begin_entity = -1

    for i in range(len(tag)):
        # Check if current tag is new entity by checking if it's 'B-' of any class
        if tag[i]%2==0 and tag[i]>=2:
            if prev_tag >=2:
                entities.append((begin_entity, i-1, prev_tag-1))
            begin_entity = i
            prev_tag = tag[i]+1
        # Check if current tag is new entity (by comparing to previous tag)
        elif tag[i] != prev_tag:
            if prev_tag >= 2:
                entities.append((begin_entity, i-1, prev_tag-1))
            begin_entity = i
            prev_tag = tag[i]
    # Check if entity continues to the end of tensor tag
    if prev_tag >= 2:
        entities.append((begin_entity, len(tag)-1, prev_tag-1))
    return entities


# def create_word2idx_dict(index_to_key, key_to_index, train_path):
#     return {w: key_to_index[w] for w in index_to_key}

# versao mais nova no ze reinaldo
def create_word2idx_dict(emb, train_path):
    return emb.key_to_index

def create_char2idx_dict(train_path, verbose=False):
    if verbose: print("train_path:", train_path)
    words = (l.split()[0] for l in open(train_path, 'r').readlines() if l != '\n')
    chars = set(''.join(words))
    dic = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
    dic.update({char: idx + 4 for (idx, char) in enumerate(chars)})
    return dic


def create_tag2idx_dict(train_path):
    lines = open(train_path, 'r').readlines()
    dic = {}
    # Each line is expected to follow the regex pattern
    # (\d+ X X \w-\w+|O)
    # or +be an empty line
    tags = set(
      [ l.split()[3][2:] for l in lines if 
        ('X X' in l and l.split('-')[-1] != 'O')
      ]
    )
    tags = {i for i in tags if i}
    iob2_dic = {'<PAD>': 0, 'O': 1}
    
    for tag in tags:
        iob2_dic[tag] = len(iob2_dic)
        for c in 'IBES':
            iob2_dic[f'{c}-{tag}'] = len(iob2_dic)

    return iob2_dic



def find_iobes_entities(sentence, tag2idx):
  t = [2+i for i in range(0, len(tag2idx), 4)]
  entities = []
  tag_flag = False
  entity_start = -1
  curr_tag_class = 0
  for i in range(len(sentence)):
    tag = sentence[i]
    # 'O' or '<PAD>' or '<GO>' classes
    if tag == 0 or tag == 1:# or tag == tag2idx['<GO>']:
      curr_tag_class = 0
      tag_flag == False
      continue
    tag_class = t[bisect(t, tag)-1]
    tag_mark  = tag - tag_class
    # B- class
    if tag_mark == 0:
      tag_flag = True
      entity_start = i
      curr_tag_class = tag_class
    # I- class
    elif tag_mark == 1 and curr_tag_class != tag_class:
      tag_flag = False
    # S- class
    elif tag_mark == 2:
      entities.append((i, i, tag_class))
      tag_flag = False
    # E- class
    elif tag_mark == 3:
      if tag_flag and (curr_tag_class == tag_class):
        entities.append((entity_start, i, tag_class))
      tag_flag = False
  return entities


def find_iobes_entities2(sentence, tag2idx):
    return set(find_iobes_entities(sentence, tag2idx))



class new_custom_collate_fn():
    def __init__(self, pad_idx, unk_idx):
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
    def __call__(self, batch):
        words = [torch.LongTensor(batch[i][0]) for i in range(len(batch))]
        tags  = [torch.LongTensor(batch[i][1]) for i in range(len(batch))]
        chars = [batch[i][2].copy() for i in range(len(batch))]

        # Pad word/tag level
        words = pad_sequence(words, batch_first = True, padding_value=self.pad_idx)
        tags  = pad_sequence(tags, batch_first = True, padding_value = 0)

        # Pad character level
        max_word_len = -1
        for sentence in chars:
            for word in sentence:
                max_word_len = max(max_word_len, len(word))
        for i in range(len(chars)):
            for j in range(len(chars[i])):
                chars[i][j] = [0 if k >= len(chars[i][j]) else chars[i][j][k] for k in range(max_word_len)]
        for i in range(len(chars)):
            chars[i] = [[0 for _ in range(max_word_len)] if j>= len(chars[i]) else chars[i][j] for j in range(words.shape[1])]
        chars = torch.LongTensor(chars)

        mask = words != self.pad_idx

        return words, tags, chars, mask

def path2str(p):
    return p if isinstance(p, str) else p.as_posix()

def load_embedding(train_path, test_path, embedding_path, embedding_size=50):
    
    print("EMBEDDING?")
    emb = KeyedVectors.load(embedding_path)
    vocab = {}
    print("train??")
    f = open(train_path)
    print("trainnnnnn")
    for line in f:
        try:
            word = line.split()[0]
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1
        except:
            pass
    not_found = {}
    for word in vocab:
        if word not in emb and word.lower() not in emb:
            not_found[word] = vocab[word]
    sorted_x = sorted(not_found.items(), key=operator.itemgetter(1))[::-1]

    # Augment pretrained embeddings with most frequent out-of-vocabulary words from the train set
    if '<START>' not in emb:
        emb['<START>'] = np.random.uniform(0.1,1,embedding_size)
    if '<END>' not in emb:
        emb['<END>'] = np.random.uniform(0.1,1,embedding_size)
    if '<UNK>' not in emb:
        emb['<UNK>'] = np.random.uniform(0.1,1,embedding_size)
    if '<PAD>' not in emb:
        emb['<PAD>'] = np.zeros(embedding_size)
    for (token, freq) in sorted_x[:37]:
        emb[token]= np.random.uniform(0.1, 1, embedding_size)

    return emb