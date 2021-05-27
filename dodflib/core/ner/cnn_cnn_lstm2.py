import torch
from torch import nn
from math import sqrt
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pad_sequence

class Char_CNN(nn.Module):
    """single layer CNN with: filters=50, kernel_size=3, dropout=0.5
    CHANGES:
    - kaiming_uniform initialization of convolution weights
    - Embedding dropout with p=0.25
    - dropout now only applied on output of convolution layers (after activation function)
    - Added weight_norm to conv layers
    """
    def __init__(self, embedding_size, embedding_dim, out_channels):
        super(Char_CNN, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=embedding_size, 
            embedding_dim=embedding_dim, 
            padding_idx=0)
        self.conv = nn.Conv1d(
            in_channels=embedding_dim, 
            out_channels=out_channels, 
            kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.init_weight()

    def init_weight(self):
        bias = sqrt(3.0/self.embedding.embedding_dim)
        nn.init.uniform_(self.embedding.weight, -bias, bias)
        # nn.init.kaiming_uniform_(self.conv.weight.data, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.dropout(self.embedding(x))
        shape = x.shape
        x = self.conv(x.reshape([shape[0]*shape[1], shape[2], shape[3]]).permute(0, 2, 1))
        x = torch.nn.functional.max_pool1d(x, kernel_size=x.shape[2]).squeeze(2)
        x = self.relu(x)
        return x.reshape([shape[0], shape[1], -1])


class Word_CNN(nn.Module):
    """
    Word-level CNN encoder
    CHANGES:
    - Now dropout only at output of forward() and relu after each conv layer
    - Added weight_norm for conv layers
    - Added embedding dropout after concat word and char embeddings
    """
    def __init__(self, pretrained_word_emb, word2idx, full_embedding_size, conv_layers, out_channels):
        super(Word_CNN, self).__init__()
        self.word2idx = word2idx
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_word_emb.vectors))
        # self.conv1 = nn.Conv1d(in_channels=full_embedding_size, out_channels=out_channels, kernel_size=5, padding=2)
        # self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(p=0.5)
        convnet = [nn.Conv1d(
                        in_channels=full_embedding_size, 
                        out_channels=out_channels, 
                        kernel_size = 5, padding = 2),
                nn.ReLU(),
                nn.Dropout(p=0.5)]

        
        for i in range(1, conv_layers):
            convnet.append(
                nn.Conv1d(
                    in_channels=out_channels, 
                    out_channels=out_channels, 
                    kernel_size = 5, padding = 2))
            convnet.append(nn.ReLU())
            convnet.append(nn.Dropout(p=0.5))

        self.convnet = nn.Sequential(*convnet)
        self.init_weight()

    def init_weight(self):
        pass
        # nn.init.kaiming_uniform_(self.conv1.weight.data, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_uniform_(self.conv2.weight.data, mode='fan_in', nonlinearity='relu')

    def forward(self, x, char_embeddings):
        # 50% chance word dropout to improve generalization
        # if self.training:
        #     mask = torch.distributions.Bernoulli(probs=(1-0.5)).sample(x.size())
        #     # x[~mask.bool()] = 314816 # glove
        #     # x[~mask.bool()] = 3000001  # googlenews
        #     x[~mask.bool()] = self.word2idx['<UNK>']
        # word2vec embedding
        x = self.embedding(x)
        # concat word and char embedding
        x = torch.cat((x, char_embeddings), dim=2)
        x = self.dropout(x)
        w = x.clone()
        x = x.permute(0, 2, 1)
        x = self.convnet(x)
        x = x.permute(0, 2, 1)
        # Return Concat w_full (word representation - combination of word and char embeddings) and h_enc (output of conv layers)
        return torch.cat((x, w), dim=2)



class LSTM_Decoder(nn.Module):
    """
    LSTM decoder (greedy decoding)
    """
    def __init__(self, feature_size, num_classes, hidden_size, decoder_layers, device):
        super(LSTM_Decoder, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.lstm = torch.nn.LSTM(
            input_size=feature_size+num_classes, 
            hidden_size=hidden_size, 
            num_layers=decoder_layers)
        self.linear = torch.nn.Linear(hidden_size, num_classes)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index = 0, reduction='sum')
        self.init_weight()

    def init_weight(self):
        # Initialize linear layer
        bias = sqrt(6.0/(self.linear.weight.shape[0]+self.linear.weight.shape[1]))
        nn.init.uniform_(self.linear.weight, -bias, bias)
        nn.init.constant_(self.linear.bias, 0.0)
        # Initialize LSTM layer
        for name, params in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(params, 0.0)
                nn.init.constant_(params[self.lstm.hidden_size:2*self.lstm.hidden_size], 1.0)
            else:
                bias = sqrt(6.0/(params.shape[0]+params.shape[1]))
                nn.init.uniform_(params, -bias, bias)

    def forward_step(self, x, prev_tag, prev_lstm_state):
        prev_tag_onehot = torch.nn.functional.one_hot(
            prev_tag, 
            num_classes=self.num_classes).to(self.device)
        lstm_input = torch.cat((x, prev_tag_onehot), dim=1).unsqueeze(dim=0)
        lstm_output, lstm_state = self.lstm(lstm_input, prev_lstm_state)
        lstm_output = self.dropout(lstm_output)
        linear_output = self.linear(lstm_output.squeeze(dim=0))
        prediction = torch.argmax(linear_output, dim=1)
        return linear_output, prediction, lstm_state
    
    def forward(self, x, tag):
        batch_size, seq_len, _ = x.size()
        x = x.permute(1, 0, 2)
        tag = tag.permute(1, 0)
        pred = torch.LongTensor([0 for i in range(batch_size)])
        lstm_state = None
        loss = 0.0
        # x = self.dropout(x)

        # Isso não deveria ser feito automaticamente pela LSTM?
        for i in range(seq_len):
            output, pred, lstm_state = self.forward_step(
                x=x[i], prev_tag=pred, prev_lstm_state=lstm_state
            )
            loss += self.loss_fn(output, tag[i])
        return loss

    def decode(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.permute(1, 0, 2)
        pred = torch.LongTensor([0 for i in range(batch_size)])
        lstm_state = None
        full_predictions = torch.LongTensor().to(self.device)
        full_output = torch.Tensor().to(self.device)
        for i in range(seq_len):
            output, pred, lstm_state = self.forward_step(x=x[i], prev_tag=pred, prev_lstm_state=lstm_state)
            full_predictions = torch.cat((full_predictions, pred.unsqueeze(dim=0)), dim=0)
            full_output = torch.cat((full_output, output.unsqueeze(dim=0)), dim=0)
        return full_predictions.permute(1, 0), full_output.permute(1, 0, 2)


class CNN_CNN_LSTM(nn.Module):
    """
    CNN-CNN-LSTM model with greedy decoding (as proposed in shen et al)
    """
    def __init__(self, char_vocab_size, char_embedding_dim, char_out_channels, word2idx, 
                    pretrained_word_emb, word_conv_layers, word_out_channels, num_classes, 
                    decoder_layers, decoder_hidden_size, device):
        super(CNN_CNN_LSTM, self).__init__()
        self.num_classes = num_classes
        self.char_encoder = Char_CNN(
            embedding_size=char_vocab_size, 
            embedding_dim=char_embedding_dim, 
            out_channels=char_out_channels)
        self.word_encoder = Word_CNN(
            pretrained_word_emb=pretrained_word_emb, 
            word2idx=word2idx, 
            conv_layers = word_conv_layers, 
            full_embedding_size=pretrained_word_emb.vector_size+char_out_channels, 
            out_channels=word_out_channels)
        self.decoder      = LSTM_Decoder(
            num_classes=num_classes, 
            feature_size=pretrained_word_emb.vector_size+char_out_channels+word_out_channels, 
            hidden_size=decoder_hidden_size, 
            decoder_layers=decoder_layers, device=device)

    def forward(self, sentence, word, tag, mask):

        x = self.char_encoder(word)
        x = self.word_encoder(sentence, x)
        x = self.decoder(x, tag)
        return x
    
    def encode(self, sentence, word, mask):
        x = self.char_encoder(word)
        x = self.word_encoder(sentence, x)
        return x

    def decode(self, sentence, word, mask, return_token_log_probabilities=False):
        x = self.char_encoder(word)
        x = self.word_encoder(sentence, x)
        predictions, output = self.decoder.decode(x)
        # Changed here (Return token log-probabilities if return_token_probabilities==True, else return sentence probability)
        if return_token_log_probabilities == True:
            return predictions, torch.nn.functional.log_softmax(output, dim=2)
        else:
            probability = torch.nn.functional.softmax(output, dim=2).max(dim=2).values
            probability[mask==False] = 1.0
            probability = probability.prod(dim = 1)
            return predictions, probability
