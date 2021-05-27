import torch
# Libs to create Dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
import itertools
import matplotlib.pyplot as plt
import re
import pandas as pd

def convert_IOB1_2_IOB2(sentence):
    prev_tag = 'O'
    for i, sent in enumerate(sentence):
        if sent[0] == 'I' and sent != prev_tag and prev_tag != 'B'+sent[1:]:
            sentence[i] = 'B' + sent[1:]
        prev_tag = sentence[i]
    return sentence


def convert_IOB2_2_IOBES(sentence):
    for i, sent in enumerate(sentence):
        if sent[0] == 'I' and (i+1==len(sentence) or sentence[i+1][0] != 'I'):
            sentence[i] = 'E' + sent[1:]
        elif sent[0] == 'B' and (i+1==len(sentence) or sentence[i+1][0] != 'I'):
            sentence[i] = 'S' + sent[1:]
    return sentence


def convert_IOB1_2_IOBES(sentence):
    return convert_IOB2_2_IOBES(convert_IOB1_2_IOB2(sentence))


def tolistify(lis):
    return (i.tolist() for i in lis)

class active_dataset(Dataset):

    @staticmethod
    def clean_numbers(x):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
        return x


    def __init__(self, word2idx_dic, char2idx_dic, tag2idx_dic,
        data, data_format='iob1', debug=False):
        # super(active_self_dataset, self).__init__()
        super(active_dataset, self).__init__()
        self.debug = debug
        self.data_format = data_format
        self.word2idx_dic = word2idx_dic
        self.char2idx_dic = char2idx_dic
        self.tag2idx_dic  = tag2idx_dic
        # Initialize labeled set
        self.labeled_sentences = []
        self.labeled_tags  = []
        self.labeled_words = []
        # Initialize unlabeled set
        self.unlabeled_sentences = []
        self.unlabeled_tags  = []
        self.unlabeled_words = []
        # Initialize full dataset
        self.sentences = []
        self.tags = []
        self.words = []

        # Flag to control labeled/unlabeled data to be retrieved by __getitem__()
        self.flag_labeled = False

        if isinstance(data, pd.DataFrame):
            aux = self.load_from_df(data, 'ato', 'iob') 
        else:
            aux = self.load_data(data)

        self.sentences, self.tags, self.words = aux

        self.char2idx()
        self.word2idx()
        self.tag2idx()

        idx = list(range(len(self.sentences)))
        self.unlabeled_sentences = idx
        self.unlabeled_tags = idx
        self.unlabeled_words = idx


    def __getitem__(self, index):
        # Change here
        if self.flag_labeled:
            return (self.sentences[self.labeled_sentences[index]], 
                    self.tags[self.labeled_tags[index]], 
                    self.words[self.labeled_words[index]])
        else:
            return (self.sentences[self.unlabeled_sentences[index]], 
                    self.tags[self.unlabeled_tags[index]], 
                    self.words[self.unlabeled_words[index]])


    def __len__(self):
        if self.flag_labeled:
            return len(self.labeled_sentences)
        else:
            return len(self.unlabeled_sentences)


    def load_from_df(self, df, text_col, labels_col, aslist=True):
        S = '<START>'
        E = '<END>'
        O = 'O'

        sentences: pd.Series = df[text_col].apply(lambda x: [S, *x, E])
        tags: pd.Series = df[labels_col].apply(lambda x: ['O', *x, 'O'])
        
        if self.data_format == 'iob1':
            tags = tags.apply(convert_IOB1_2_IOBES)
        elif self.data_format == 'iob2':
            tags = tags.apply(convert_IOB2_2_IOBES)
            
        words: pd.Series = df[text_col].apply(lambda x: [[S, *el, E] for el in x])
        if aslist:
            return sentences.tolist(), tags.tolist(), words.tolist()
        else:
            return sentences, tags, words


    def load_data(self, path, ):
        f = open(path, 'r').readlines()

        temp_sentences = []
        temp_tags = []
        temp_words = []

        sentences = []
        tags = []
        words = []

        for line in f:
            if line == '\n' or not line:
                if temp_sentences:
                    temp_sentences = ['<START>', *temp_sentences, '<END>']
                    temp_tags = ['O', *temp_tags, 'O']
                    temp_words = [['<START>', *word, '<END>'] for word in temp_sentences]
                    
                    if self.debug:
                        print('\ttemp_sentences:', temp_sentences)
                        print('\ttemp_tags:', temp_tags)
                        print('\ttemp_words:', temp_words)

                        input()

                    sentences.append(temp_sentences)
                    words.append(temp_words)

                    if self.data_format == 'iob1':
                        tags.append(convert_IOB2_2_IOBES(convert_IOB1_2_IOB2(temp_tags)))
                    elif self.data_format == 'iob2':
                        tags.append(convert_IOB2_2_IOBES(temp_tags))
                    else:
                        tags.append(temp_tags)

                    temp_sentences = []
                    temp_tags = []
                    temp_words = []

            else:
                # Precisa mudar apenas essa parte:
                temp_sentences.append(line.split()[0])
                temp_tags.append(line.split()[3])
                
        if temp_sentences:
            temp_sentences = ['<START>', *temp_sentences, '<END>']
            temp_tags = ['O', *temp_tags, 'O']
            temp_words = [
                [*itertools.chain(['<START>'], [*word], ['<END>'])]
                for word in temp_sentences
            ]
            sentences.append(temp_sentences)
            words.append(temp_words)
            if self.data_format == 'iob1':
                tags.append(self.convert_IOB2_2_IOBES(self.convert_IOB1_2_IOB2(temp_tags)))
            elif self.data_format == 'iob2':
                tags.append(self.convert_IOB2_2_IOBES(temp_tags))
            else:
                tags.append(temp_tags)

        return sentences, tags, words
    

    def word2idx(self):
        for i, sent in enumerate(self.sentences):
            for j, dirty_word in enumerate(sent):
                word = self.clean_numbers(dirty_word)
                if word in self.word2idx_dic:
                    self.sentences[i][j] = self.word2idx_dic[word]
                elif word.lower() in self.word2idx_dic:
                    self.sentences[i][j] = self.word2idx_dic[word.lower()]
                else:
                    self.sentences[i][j] = self.word2idx_dic['<UNK>']


    def tag2idx(self):
        for i, tag_lis in enumerate(self.tags):
            for j, tag in enumerate(tag_lis):
                self.tags[i][j] = self.tag2idx_dic[tag]

    def char2idx(self):
        UNKNOW = self.char2idx_dic.get('<UNK>')
        self.words = [
            [
                [
                    self.char2idx_dic.get(char, UNKNOW) for char in word]
                    for word in sentence
                ]
                for sentence in self.words
        ]


    def label_data(self, idx_list):
        """Sends data samples from unlabeled set to the labeled set"""
        
        # new_unlabeled_sentences = [self.unlabeled_sentences[i] for i in range(len(self.unlabeled_sentences)) if i not in idx_list]
        new_unlabeled_sentences = [usent for (i, usent) in enumerate(self.unlabeled_sentences) if i not in idx_list]
        # new_unlabeled_tags = [self.unlabeled_tags[i] for i in range(len(self.unlabeled_tags)) if i not in idx_list]
        new_unlabeled_tags = [utags for (i, utags) in enumerate(self.unlabeled_tags) if i not in idx_list]
        # new_unlabeled_words = [self.unlabeled_words[i] for i in range(len(self.unlabeled_words)) if i not in idx_list]
        new_unlabeled_words = [uwords for (i, uwords) in enumerate(self.unlabeled_words) if i not in idx_list]
        
        new_labeled_sentences = [self.unlabeled_sentences[i] for i in idx_list]
        new_labeled_tags = [self.unlabeled_tags[i] for i in idx_list]
        new_labeled_words = [self.unlabeled_words[i] for i in idx_list]
        
        # self.unlabeled_sentences = new_unlabeled_sentences
        # self.unlabeled_tags = new_unlabeled_tags
        # self.unlabeled_words = new_unlabeled_words


        self.labeled_sentences += new_labeled_sentences
        self.labeled_tags += new_labeled_tags
        self.labeled_words += new_labeled_words
        
        # Sort sentences by sentence length
        # self.unlabeled_sentences, self.unlabeled_words, self.unlabeled_tags = self.sort_set(self.unlabeled_sentences, self.unlabeled_words, self.unlabeled_tags)
        self.unlabeled_sentences, self.unlabeled_words, self.unlabeled_tags = self.sort_set(
            new_unlabeled_sentences, new_unlabeled_words, new_unlabeled_tags
        )
        self.labeled_sentences, self.labeled_words, self.labeled_tags = self.sort_set(self.labeled_sentences, self.labeled_words, self.labeled_tags)


    def sort_set(self, unordered_sentences, unordered_words, unordered_tags):
        # Change here
        ordered_idx = np.argsort(
            # [len(self.sentences[unordered_sentences[i]]) for i in range(len(unordered_sentences))]
            [len(self.sentences[usent]) for usent in unordered_sentences]
        )
        # TODO: colocar "index-map" em funcao
        ordered_sentences = [unordered_sentences[i] for i in ordered_idx]
        ordered_words = [unordered_words[i] for i in ordered_idx]
        ordered_tags = [unordered_tags[i] for i in ordered_idx]
        return ordered_sentences, ordered_words, ordered_tags


    def get_word_count(self):
        # Change here
        if self.flag_labeled:
            sentences_idx = self.labeled_sentences
        else:
            sentences_idx = self.unlabeled_sentences
        word_count = 0
        for idx in sentences_idx:
            sentence = self.sentences[idx]
            # word_count += len(sentence) - 2
            for word in sentence:
                if word != self.word2idx_dic['<PAD>'] and word != self.word2idx_dic['<START>'] and word != self.word2idx_dic['<END>']:
                    word_count += 1
        return word_count



# class active_self_dataset(Dataset):
class active_self_dataset(active_dataset):

    def __init__(self, word2idx_dic, char2idx_dic, tag2idx_dic,
        path, data_format='iob1'):
        super(active_self_dataset, self).__init__()
        self.data_format = data_format
        self.word2idx_dic = word2idx_dic
        self.char2idx_dic = char2idx_dic
        self.tag2idx_dic  = tag2idx_dic
        # Initialize active labeled set
        self.active_labeled_set = []
        # Initialize self labeled set
        self.self_labeled_set = []
        # Initialize unlabeled set
        self.unlabeled_set = []
        # Initialize full dataset
        self.sentences = []
        self.tags = []
        self.words = []

        # Flag to control active_labeled/self_labeled/unlabeled data to be retrieved by __getitem__()
        self.flag_labeled = 'unlabeled'

        self.sentences, self.tags, self.words = self.load_data(path)

        self.char2idx()
        self.word2idx()
        self.tag2idx()

        idx = [i for i in range(len(self.sentences))]
        self.unlabeled_set = idx

    def __getitem__(self, index):
        
        if self.flag_labeled == 'active_labeled':
            return self.sentences[self.active_labeled_set[index]], self.tags[self.active_labeled_set[index]], self.words[self.active_labeled_set[index]]
        elif self.flag_labeled == 'self_labeled':
            return self.sentences[self.self_labeled_set[index]], self.tags[self.self_labeled_set[index]], self.words[self.self_labeled_set[index]]
        elif self.flag_labeled == 'unlabeled':
            return self.sentences[self.unlabeled_set[index]], self.tags[self.unlabeled_set[index]], self.words[self.unlabeled_set[index]]

    def __len__(self):

        if self.flag_labeled == 'active_labeled':
            return len(self.active_labeled_set)
        elif self.flag_labeled == 'self_labeled':
            return len(self.self_labeled_set)
        elif self.flag_labeled == 'unlabeled':
            return len(self.unlabeled_set)


    def self_label_data(self, idx_list):
        """Switches data samples from unlabeled set to self labeled set"""

        new_unlabeled_samples = [self.unlabeled_set[i] for i in range(len(self.unlabeled_set)) if i not in idx_list]
        new_labeled_samples = [self.unlabeled_set[i] for i in idx_list]
        
        self.unlabeled_set = new_unlabeled_samples
        self.self_labeled_set += new_labeled_samples
        
        # Sort sentences by sentence length
        self.unlabeled_set = self.sort_set(self.unlabeled_set)
        self.self_labeled_set = self.sort_set(self.self_labeled_set)

    def active_label_data(self, idx_list):
        """Switches data samples from unlabeled set to active labeled set"""

        new_unlabeled_samples = [self.unlabeled_set[i] for i in range(len(self.unlabeled_set)) if i not in idx_list]
        new_labeled_samples = [self.unlabeled_set[i] for i in idx_list]
        
        self.unlabeled_set = new_unlabeled_samples
        self.active_labeled_set += new_labeled_samples
        
        # Sort sentences by sentence length
        self.unlabeled_set = self.sort_set(self.unlabeled_set)
        self.active_labeled_set = self.sort_set(self.active_labeled_set)

    def sort_set(self, unordered_set):
        # Sort set by sentence length
        ordered_idx = np.argsort([len(self.sentences[unordered_set[i]]) for i in range(len(unordered_set))])
        return [unordered_set[i] for i in ordered_idx]

    def get_word_count(self):
        """
        Returns total number of tokens in one of the datasets (active_labeled, self_labeled, unlabeled)
        """
        if self.flag_labeled == 'active_labeled':
            sentences_idx = self.active_labeled_set
        elif self.flag_labeled == 'self_labeled':
            sentences_idx = self.self_labeled_set
        elif self.flag_labeled == 'unlabeled':
            sentences_idx = self.unlabeled_set
        word_count = 0
        for idx in sentences_idx:
            sentence = self.sentences[idx]
            for word in sentence:
                if word != self.word2idx_dic['<PAD>'] and word != self.word2idx_dic['<START>'] and word != self.word2idx_dic['<END>']:
                    word_count += 1
        return word_count


