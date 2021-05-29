import os
import math
import pandas as pd
import numpy as np
import argparse

from pathlib import Path
from typing import Iterable as Iter
from sklearn.base import BaseEstimator, TransformerMixin


import torch
from seqeval.metrics import f1_score
from seqeval.scheme import IOBES

from gensim.models import KeyedVectors
from torch.utils.data import Dataset, DataLoader

try:

    from dodflib.core.ner.CNN_biLSTM_CRF import CNN_biLSTM_CRF as M1
    from dodflib.core.ner.CNN_CNN_LSTM import CNN_CNN_LSTM as M2
    from dodflib.core.ner import utils
    from dodflib.core.ner import metrics
    from dodflib.core.ner import data3

except:
    from CNN_biLSTM_CRF import CNN_biLSTM_CRF as M1
    from CNN_CNN_LSTM import CNN_CNN_LSTM as M2
    import utils
    import metrics
    import data3
    


def path2str(p):
    return p if isinstance(p, str) else p.as_posix()


def parse_ner_args():
    p = argparse.ArgumentParser(description='Supervised training procedure for NER models!')
    p.add_argument('--save_training_path', dest='save_training', type=str, default=None, help='Path to save training history and hyperaparms used')
    p.add_argument('--save_model_path', dest='save_model', type=str, default=None, help='Path to save trained model')
    p.add_argument('--epochs', dest='epochs', action='store', type=int, default=50, help='Number of supervised training epochs')
    # dataset parameters
    p.add_argument('--train_path', dest='train_path', action='store', type=str, default=None, help='Path to load training set from')
    p.add_argument('--test_path', dest='test_path', action='store', type=str, default=None, help='Path to load testing set from')
    p.add_argument('--dataset_format', dest='dataset_format', action='store', type=str, default='iob2', help='Format of the dataset (e.g. iob1, iob2, iobes)')
    # Embedding parameters
    p.add_argument('--embedding_path', dest='embedding_path', type=str, default=None, help='Path to load pretrained embeddings from')
    p.add_argument('--augment_pretrained_embedding', dest='augment_pretrained_embedding', type=bool, default=False, help='Indicates whether to augment pretrained embeddings with vocab from training set')
    # General model parameters
    p.add_argument('--model', dest='model', action='store', type=str, default='CNN-biLSTM-CRF', help='Neural NER model architecture')
    p.add_argument('--char_embedding_dim', dest='char_embedding_dim', action='store', type=int, default=30, help='Embedding dimension for each character')
    p.add_argument('--char_out_channels', dest='char_out_channels', action='store', type=int, default=50, help='# of channels to be used in 1-d convolutions to form character level word embeddings')
    # CNN-CNN-LSTM specific parameters
    p.add_argument('--word_out_channels', dest='word_out_channels', action='store', type=int, default=800, help='# of channels to be used in 1-d convolutions to encode word-level features')
    p.add_argument('--word_conv_layers', dest='word_conv_layers', action='store', type=int, default=2, help='# of convolution blocks to be used to encode word-level features')
    p.add_argument('--decoder_layers', dest='decoder_layers', action='store', type=int, default=1, help='# of layers of the LSTM greedy decoder')
    p.add_argument('--decoder_hidden_size', dest='decoder_hidden_size', action='store', type=int, default=256, help='Size of the LSTM greedy decoder layer')
    # CNN-biLSTM-CRF specific parameters
    p.add_argument('--lstm_hidden_size', dest='lstm_hidden_size', action='store', type=int, default=200, help='Size of the lstm for word-level feature encoder')
    # Trainign hyperparameters
    p.add_argument('--lr', dest='lr', action='store', type=float, default=0.0015, help='Learning rate for NER mdoel training')
    p.add_argument('--clipping_value', dest='clipping_value', action='store', type=float, default=5.0, help='Value at which to clip the model gradient throughout training')
    p.add_argument('--momentum', dest='momentum', action='store', type=float, default=0.9, help='Momentum for the SGD optimization process')
    # Training parameters
    p.add_argument('--batch_size', dest='batch_size', action='store', type=int, default=16, help='Batch size for training')

    args, _ = p.parse_known_args()
    return args


class CNN_biLSTM_CRF(BaseEstimator, TransformerMixin):

    def __init__(self, char_embedding_dim,  char_out_channels, embedding_path, lstm_hidden_size,
                 train_path, test_path, augment_pretrained_embedding=False, dataset_format='iob2',
                    **kwargs):
        
        self.train_path = train_path
        self.test_path = test_path
        self.dataset_format = dataset_format

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm_hidden_size = lstm_hidden_size
        self.emb = utils.load_embedding(train_path, test_path, path2str(embedding_path))
        # if augment_pretrained_embedding:
        #     utils.augment_pretrained_embedding(self.emb, self.train_path)
        
        bias = math.sqrt(3/self.emb.vector_size)
        if '<START>' not in self.emb:
            self.emb['<START>'] = np.random.uniform(-bias, bias, self.emb.vector_size)
        if '<END>' not in self.emb:
            self.emb['<END>'] = np.random.uniform(-bias, bias, self.emb.vector_size)
        if '<UNK>' not in self.emb:
            self.emb['<UNK>'] = np.random.uniform(-bias, bias, self.emb.vector_size)
        if '<PAD>' not in self.emb:
            self.emb['<PAD>'] = np.zeros(self.emb.vector_size)

        self.collate_object = utils.new_custom_collate_fn(
            pad_idx=self.emb.key_to_index['<PAD>'], unk_idx=self.emb.key_to_index['<UNK>']
        )
        self.word2idx = utils.create_word2idx_dict(self.emb, self.train_path)
        self.char2idx = utils.create_char2idx_dict(train_path=self.train_path)
        self.tag2idx  = utils.create_tag2idx_dict(train_path=self.train_path)

        self.model = M1(
            char_embedding_dim=char_embedding_dim,
            char_out_channels=char_out_channels,
            pretrained_word_emb=self.emb,
            char_vocab_size=len(self.char2idx),
            num_classes=len(self.tag2idx),
            device=self.device,

            lstm_hidden_size = lstm_hidden_size,
        )
        self.model = self.model.to(self.device)


    def fit(self, test_path, test_format, batch_size, lr, clipping_value, momentum, epochs, **kwargs):

        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        train_set = data3.active_dataset(
            data=self.train_path, word2idx_dic=self.word2idx, char2idx_dic=self.char2idx, 
            tag2idx_dic=self.tag2idx, data_format=self.dataset_format)
        train_dataloader = DataLoader(
            train_set, batch_size=batch_size, pin_memory=True, 
            collate_fn = self.collate_object, shuffle=False)

        test_set  = data3.active_dataset(
            data=self.test_path, word2idx_dic=self.word2idx, char2idx_dic=self.char2idx, 
            tag2idx_dic=self.tag2idx, data_format=self.dataset_format)
        test_dataloader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False, collate_fn=self.collate_object)

        f1_history = []
        for epoch in range(epochs):
            print("Epoch:", epoch)
            self.model.train()
            for sent, tag, word, mask in train_dataloader:
                sent = sent.to(self.device)
                tag = tag.to(self.device)
                word = word.to(self.device)
                mask = mask.to(self.device)
                optim.zero_grad()
                loss = self.model(sent, word, tag, mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clipping_value)
                optim.step()
            
            self.model.eval()
            with torch.no_grad():
                predictions, targets = metrics.preprocess_pred_targ(
                    self.model, test_dataloader, self.device)
                predictions = metrics.IOBES_tags(predictions, self.tag2idx)
                targets = metrics.IOBES_tags(targets, self.tag2idx)
                micro_f1 = f1_score(targets, predictions, mode='strict', scheme=IOBES)
                f1_history.append(0 if np.isnan(micro_f1) else micro_f1)
                print(f'micro f1-score: {micro_f1}\n')


    def predict(self, test_path: str, batch_size: int):
        test_set  = data3.active_dataset(
            data=test_path, word2idx_dic=self.word2idx, char2idx_dic=self.char2idx, 
            tag2idx_dic=self.tag2idx, data_format=self.dataset_format)
        test_dataloader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False, collate_fn=self.collate_object)

        with torch.no_grad():
            predictions, targets = metrics.preprocess_pred_targ(
                self.model, test_dataloader, self.device)
            predictions = metrics.IOBES_tags(predictions, self.tag2idx)
            targets = metrics.IOBES_tags(targets, self.tag2idx)
        return predictions


class CNN_CNN_LSTM(BaseEstimator, TransformerMixin):
    # def __init__(self, char_vocab_size, char_embedding_dim, char_out_channels, 
    #     pretrained_word_emb, word2idx, word_out_channels, word_conv_layers, 
    #     num_classes, decoder_layers, decoder_hidden_size, device, ):
    def __init__(self, char_embedding_dim,  char_out_channels, embedding_path, train_path,
                    test_path, word_out_channels, word_conv_layers, decoder_layers, 
                    decoder_hidden_size, augment_pretrained_embedding=False, dataset_format='iob2',
                    **kwargs):

        self.train_path = train_path
        self.test_path = test_path
        self.dataset_format = dataset_format

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.emb = utils.load_embedding(train_path, test_path, path2str(embedding_path))
        # if augment_pretrained_embedding:
        #     utils.augment_pretrained_embedding(self.emb, self.train_path)
        
        bias = math.sqrt(3/self.emb.vector_size)
        if '<START>' not in self.emb:
            self.emb['<START>'] = np.random.uniform(-bias, bias, self.emb.vector_size)
        if '<END>' not in self.emb:
            self.emb['<END>'] = np.random.uniform(-bias, bias, self.emb.vector_size)
        if '<UNK>' not in self.emb:
            self.emb['<UNK>'] = np.random.uniform(-bias, bias, self.emb.vector_size)
        if '<PAD>' not in self.emb:
            self.emb['<PAD>'] = np.zeros(self.emb.vector_size)

        self.collate_object = utils.new_custom_collate_fn(
            pad_idx=self.emb.key_to_index['<PAD>'], unk_idx=self.emb.key_to_index['<UNK>']
        )

        self.word2idx = utils.create_word2idx_dict(self.emb, self.train_path)
        self.char2idx = utils.create_char2idx_dict(train_path=self.train_path)
        self.tag2idx  = utils.create_tag2idx_dict(train_path=self.train_path)

        self.model = M2(
            char_embedding_dim=char_embedding_dim,
            char_out_channels=char_out_channels,
            device=self.device,
            pretrained_word_emb=self.emb,
            char_vocab_size=len(self.char2idx),
            num_classes=len(self.tag2idx),
            word2idx=self.word2idx,

            word_out_channels = word_out_channels,
            word_conv_layers = word_conv_layers,
            decoder_layers = decoder_layers,
            decoder_hidden_size = decoder_hidden_size,
        )
        self.model = self.model.to(self.device)


    def fit(self, test_path, test_format, batch_size, lr, clipping_value, momentum, epochs, **kwargs):

        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        train_set = data3.active_dataset(
            data=self.train_path, word2idx_dic=self.word2idx, char2idx_dic=self.char2idx, 
            tag2idx_dic=self.tag2idx, data_format=self.dataset_format)
        train_dataloader = DataLoader(
            train_set, batch_size=batch_size, pin_memory=True, 
            collate_fn = self.collate_object, shuffle=False)

        test_set  = data3.active_dataset(
            data=self.test_path, word2idx_dic=self.word2idx, char2idx_dic=self.char2idx, 
            tag2idx_dic=self.tag2idx, data_format=self.dataset_format)
        test_dataloader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False, collate_fn=self.collate_object)

        f1_history = []
        for epoch in range(epochs):
            print("Epoch:", epoch)
            self.model.train()
            for sent, tag, word, mask in train_dataloader:
                sent = sent.to(self.device)
                tag = tag.to(self.device)
                word = word.to(self.device)
                mask = mask.to(self.device)
                optim.zero_grad()
                loss = self.model(sent, word, tag, mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clipping_value)
                optim.step()
            
            self.model.eval()
            with torch.no_grad():
                predictions, targets = metrics.preprocess_pred_targ(
                    self.model, test_dataloader, self.device)
                predictions = metrics.IOBES_tags(predictions, self.tag2idx)
                targets = metrics.IOBES_tags(targets, self.tag2idx)
                micro_f1 = f1_score(targets, predictions, mode='strict', scheme=IOBES)
                f1_history.append(0 if np.isnan(micro_f1) else micro_f1)
                print(f'micro f1-score: {micro_f1}\n')

        
    def predict(self, test_path: str, batch_size: int):
        test_set  = data3.active_dataset(
            data=test_path, word2idx_dic=self.word2idx, char2idx_dic=self.char2idx, 
            tag2idx_dic=self.tag2idx, data_format=self.dataset_format)
        test_dataloader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False, collate_fn=self.collate_object)

        with torch.no_grad():
            predictions, targets = metrics.preprocess_pred_targ(
                self.model, test_dataloader, self.device)
            predictions = metrics.IOBES_tags(predictions, self.tag2idx)
            targets = metrics.IOBES_tags(targets, self.tag2idx)
        return predictions
