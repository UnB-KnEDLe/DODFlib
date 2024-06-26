# Basic packages
import argparse
import numpy as np
from gensim.models import KeyedVectors
import torch
from torch.utils.data import DataLoader
import itertools
import joblib
import re
import matplotlib.pyplot as plt
import operator
import math
from math import sqrt
from random import sample
# NER open packages
from seqeval.scheme import IOBES
from seqeval.metrics import f1_score
# my NER packages
from data import dataset
from utils import create_char2idx_dict, create_tag2idx_dict, create_word2idx_dict, new_custom_collate_fn, budget_limit, augment_pretrained_embedding
from metrics import preprocess_pred_targ, IOBES_tags
from CNN_biLSTM_CRF import cnn_bilstm_crf
from CNN_CNN_LSTM import CNN_CNN_LSTM

#| ToDo list
# (DONE) Transform code from function to script
# (DONE) Instantiate model with parameters from user input (instead of using fixed parameters depending on dataset)
# (DONE) Load embedding from path given by user or create new embedding layer if None is passed as path
# (DONE) Load dataset from path given (not from predetermined paths)
# - Describe all functions and methods including input and output type and shape
# - Load embedding from .txt files - only .kv files accepted so far :( 
# (DONE) Create requirements file

parser = argparse.ArgumentParser(description='Supervised training procedure for NER models!')
parser.add_argument('--save_training_path', dest='save_training', type=str, default=None, help='Path to save training history and hyperaparms used')
parser.add_argument('--save_model_path', dest='save_model', type=str, default=None, help='Path to save trained model')
parser.add_argument('--epochs', dest='epochs', action='store', type=int, default=50, help='Number of supervised training epochs')
# dataset parameters
parser.add_argument('--train_path', dest='train_path', action='store', type=str, default=None, help='Path to load training set from')
parser.add_argument('--test_path', dest='test_path', action='store', type=str, default=None, help='Path to load testing set from')
parser.add_argument('--dataset_format', dest='dataset_format', action='store', type=str, default='iob1', help='Format of the dataset (e.g. iob1, iob2, iobes)')
# Embedding parameters
parser.add_argument('--embedding_path', dest='embedding_path', type=str, default=None, help='Path to load pretrained embeddings from')
parser.add_argument('--augment_pretrained_embedding', dest='augment_pretrained_embedding', type=bool, default=False, help='Indicates whether to augment pretrained embeddings with vocab from training set')
# General model parameters
parser.add_argument('--model', dest='model', action='store', type=str, default='CNN-biLSTM-CRF', help='Neural NER model architecture')
parser.add_argument('--char_embedding_dim', dest='char_embedding_dim', action='store', type=int, default=30, help='Embedding dimension for each character')
parser.add_argument('--char_out_channels', dest='char_out_channels', action='store', type=int, default=50, help='# of channels to be used in 1-d convolutions to form character level word embeddings')
# CNN-CNN-LSTM specific parameters
parser.add_argument('--word_out_channels', dest='word_out_channels', action='store', type=int, default=800, help='# of channels to be used in 1-d convolutions to encode word-level features')
parser.add_argument('--word_conv_layers', dest='word_conv_layers', action='store', type=int, default=2, help='# of convolution blocks to be used to encode word-level features')
parser.add_argument('--decoder_layers', dest='decoder_layers', action='store', type=int, default=1, help='# of layers of the LSTM greedy decoder')
parser.add_argument('--decoder_hidden_size', dest='decoder_hidden_size', action='store', type=int, default=256, help='Size of the LSTM greedy decoder layer')
# CNN-biLSTM-CRF specific parameters
parser.add_argument('--lstm_hidden_size', dest='lstm_hidden_size', action='store', type=int, default=200, help='Size of the lstm for word-level feature encoder')
# Trainign hyperparameters
parser.add_argument('--lr', dest='lr', action='store', type=float, default=0.0015, help='Learning rate for NER mdoel training')
parser.add_argument('--grad_clip', dest='grad_clip', action='store', type=float, default=5.0, help='Value at which to clip the model gradient throughout training')
parser.add_argument('--momentum', dest='momentum', action='store', type=float, default=0.9, help='Momentum for the SGD optimization process')

# Training parameters
parser.add_argument('--batch_size', dest='batch_size', action='store', type=int, default=16, help='Batch size for training')
parser_opt = parser.parse_args()

print(f'\n****************************************************************************************************************')
print(f'****************************************************************************************************************')
print(f'****************************************************************************************************************')
print(f'Experiment: Supervised training')
print(f'Model: {parser_opt.model}')
print(f'batch size: {parser_opt.batch_size}')
print(f'Hardware available: {"cuda" if torch.cuda.is_available() else "cpu"}')
# ==============================================================================================
# ==============================================================================================
# =============================     Load embeddings     ========================================
# ==============================================================================================
# ==============================================================================================

emb = KeyedVectors.load(parser_opt.embedding_path)
if parser_opt.augment_pretrained_embedding:
    augment_pretrained_embedding(emb, parser_opt.train_path)

bias = sqrt(3/emb.vector_size)
if '<START>' not in emb:
    emb.add('<START>', np.random.uniform(-bias, bias, emb.vector_size))
if '<END>' not in emb:
    emb.add('<END>', np.random.uniform(-bias, bias, emb.vector_size))
if '<UNK>' not in emb:
    emb.add('<UNK>', np.random.uniform(-bias, bias, emb.vector_size))
if '<PAD>' not in emb:
    emb.add('<PAD>', np.zeros(100))

# ==============================================================================================
# ==============================================================================================q'
# ============================ Create train and test sets ======================================
# ==============================================================================================
# ==============================================================================================

collate_object = new_custom_collate_fn(pad_idx=emb.key_to_index['<PAD>'], unk_idx=emb.key_to_index['<UNK>'])

print('\nGenerating text2idx dictionaries (word, char, tag)')
word2idx = create_word2idx_dict(emb, parser_opt.train_path)
char2idx = create_char2idx_dict(train_path=parser_opt.train_path)
tag2idx  = create_tag2idx_dict(train_path=parser_opt.train_path)

print('\nCreating training dataset')
train_set = dataset(path=parser_opt.train_path, word2idx_dic=word2idx, char2idx_dic=char2idx, tag2idx_dic=tag2idx, data_format=parser_opt.dataset_format)

print('\nCreating test dataset')
test_set  = dataset(path=parser_opt.test_path, word2idx_dic=word2idx, char2idx_dic=char2idx, tag2idx_dic=tag2idx, data_format=parser_opt.dataset_format)
test_dataloader = DataLoader(test_set, batch_size=128, shuffle=False, collate_fn=collate_object)

# ==============================================================================================
# ==============================================================================================
# ============================= Instantiate neural model =======================================
# ==============================================================================================
# ==============================================================================================

# Instantiating the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if parser_opt.model == 'CNN-CNN-LSTM':
    model = CNN_CNN_LSTM(
        char_vocab_size=len(char2idx),
        char_embedding_dim=parser_opt.char_embedding_dim,
        char_out_channels=parser_opt.char_out_channels,
        pretrained_word_emb=emb,
        word2idx = word2idx,
        word_out_channels=parser_opt.word_out_channels,
        word_conv_layers = parser_opt.word_conv_layers,
        num_classes=len(tag2idx),
        decoder_layers = parser_opt.decoder_layers,
        decoder_hidden_size = parser_opt.decoder_hidden_size,
        device=device
    )
elif parser_opt.model == 'CNN-biLSTM-CRF':
    model = cnn_bilstm_crf(
        char_vocab_size=len(char2idx),
        char_embedding_dim=parser_opt.char_embedding_dim,
        char_out_channels=parser_opt.char_out_channels,
        pretrained_word_emb=emb,
        lstm_hidden_size=parser_opt.lstm_hidden_size,
        num_classes=len(tag2idx),
        device=device,
    )

lrate = parser_opt.lr
clipping_value = parser_opt.grad_clip
momentum = parser_opt.momentum

model = model.to(device)

# ==============================================================================================
# ==============================================================================================
# =============================== Define training hyperparams ==================================
# ==============================================================================================
# ==============================================================================================

# Defining supervised training hyperparameters
supervised_epochs = parser_opt.epochs
optim = torch.optim.SGD(model.parameters(), lr=lrate, momentum=momentum)

# ==============================================================================================
# ==============================================================================================
# ============================= Supervised learning algorithm ==================================
# ==============================================================================================
# ==============================================================================================
print(f'\nInitiating supervised training\n\n')
f1_history = []

dataloader = DataLoader(train_set, batch_size=parser_opt.batch_size, pin_memory=True, collate_fn = collate_object, shuffle=False)

# Supervised training (traditional)    
for epoch in range(supervised_epochs):
    print(f'Epoch: {epoch}')

    # Supervised training for one epoch
    model.train()
    for sent, tag, word, mask in dataloader:
        sent = sent.to(device)
        tag = tag.to(device)
        word = word.to(device)
        mask = mask.to(device)
        optim.zero_grad()
        loss = model(sent, word, tag, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
        optim.step()
    
    # Verify performance on test set after supervised training epoch
    model.eval()
    with torch.no_grad():
        predictions, targets = preprocess_pred_targ(model, test_dataloader, device)
        predictions = IOBES_tags(predictions, tag2idx)
        targets = IOBES_tags(targets, tag2idx)
        micro_f1 = f1_score(targets, predictions, mode='strict', scheme=IOBES)
        f1_history.append(0 if np.isnan(micro_f1) else micro_f1)
        print(f'micro f1-score: {micro_f1}\n')

# ==============================================================================================
# ==============================================================================================
# ================================= Save training history ======================================
# ==============================================================================================
# ==============================================================================================

hyperparams = {'model': str(model), 'LR': lrate, 'momentum': momentum, 'clipping': clipping_value}
dic = {'f1_hist': f1_history, 'hyperparams': hyperparams}
if parser_opt.save_training:
    joblib.dump(dic, parser_opt.save_training)
if parser_opt.save_model:
    torch.save(model.state_dict(), parser_opt.save_model)

