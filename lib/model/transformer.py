import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers, dropout_prob, init_drop=False):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim    
        layers = [nn.Linear(in_dim, hidden_layers[0]), nn.ReLU()] 
        layers += [nn.Dropout(dropout_prob)] if init_drop else []
        for i in range(1, len(hidden_layers)):
            layers.extend([nn.Linear(hidden_layers[i-1], hidden_layers[i]), nn.ReLU(), nn.Dropout(dropout_prob)])
        layers.append(nn.Linear(hidden_layers[-1], out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x, with_uncertainty=False):
        return self.model(x)

class GFNTransformer(nn.Module):
    def __init__(self, num_hid, max_len, vocab_size, num_actions, dropout, num_layers, partition_init,
                num_head, **kwargs):
        super().__init__()
        self.pos = PositionalEncoding(num_hid, dropout=dropout, max_len=max_len + 2)
        self.embedding = nn.Embedding(vocab_size, num_hid)
        encoder_layers = nn.TransformerEncoderLayer(num_hid, num_head, num_hid, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        # self.output = nn.Linear(num_hid + num_hid, num_actions)
        self.output = MLP(num_hid, num_actions, [4 * num_hid, 4 * num_hid], dropout)
        self.Z_mod = nn.Parameter(torch.ones(num_hid) * partition_init / num_hid)
        # self.Z_mod = MLP(cond_dim, num_hid, [num_hid, num_hid], 0.05)
        self.logsoftmax2 = torch.nn.LogSoftmax(2)
        self.num_hid = num_hid

    def Z(self):
        return self.Z_mod.sum()

    def model_params(self):
        return list(self.pos.parameters()) + list(self.embedding.parameters()) + list(self.encoder.parameters()) + \
            list(self.output.parameters())

    def Z_param(self):
        return [self.Z_mod]

    def forward(self, x, mask, return_all=False, lens=None, logsoftmax=False):
        x = self.embedding(x)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=mask,
                            mask=generate_square_subsequent_mask(x.shape[0]).to(x.device))
        pooled_x = x[lens-1, torch.arange(x.shape[1])]

        # cond_var = self.cond_embed(cond) # batch x hidden_dim
        # cond_var = torch.tile(cond_var, (x.shape[0], 1, 1)) if return_all else cond_var
        
        if return_all:
            out = self.output(x)
            return self.logsoftmax2(out) if logsoftmax else out
        
        y = self.output(pooled_x)
        return y

class Transformer(nn.Module):
    def __init__(self, num_hid, max_len, vocab_size, num_outputs, dropout, num_layers,
                num_head, **kwargs):
        super().__init__()
        self.pos = PositionalEncoding(num_hid, dropout=dropout, max_len=max_len + 2)
        self.embedding = nn.Embedding(vocab_size, num_hid)
        encoder_layers = nn.TransformerEncoderLayer(num_hid, num_head, num_hid, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output = MLP(num_hid, num_outputs, [4 * num_hid, 4 * num_hid], dropout)
        self.logsoftmax2 = torch.nn.LogSoftmax(2)
        self.num_hid = num_hid

    def model_params(self):
        return list(self.pos.parameters()) + list(self.embedding.parameters()) + list(self.encoder.parameters()) + \
            list(self.output.parameters())

    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        pooled_x = x[0, torch.arange(x.shape[1])]

        y = self.output(pooled_x)
        return y


def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)