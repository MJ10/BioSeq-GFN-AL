import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, num_tokens, num_outputs, num_hid,
                 num_layers, max_len=60, dropout=0.1,
                 partition_init=150.0,
                 **kwargs):
        super().__init__()
        self.input = nn.Linear(num_tokens * max_len, num_hid)
        hidden_layers = []
        for _ in range(num_layers):
            hidden_layers.append(nn.Dropout(dropout))
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.Linear(num_hid, num_hid))
        self.hidden = nn.Sequential(*hidden_layers)
        self.output = nn.Linear(num_hid, num_outputs)
        self.max_len = max_len
        self.num_tokens = num_tokens
        self._Z = nn.Parameter(torch.ones(64) * partition_init / 64)

    @property
    def Z(self):
        return self._Z.sum()

    def model_params(self):
        return list(self.input.parameters()) + list(self.hidden.parameters()) + list(self.output.parameters())

    def Z_param(self):
        return [self._Z]

    def forward(self, x, mask, return_all=False, lens=None):
        if return_all:
            outputs = []
            for i in range(lens[0]):
                mask = torch.cat((torch.ones(x.shape[0], self.num_tokens * i), torch.zeros(x.shape[0], self.num_tokens * (lens[0] - i))), axis=1)
                mask = mask.to(x.device)
                masked_input = mask * x
                out = self.input(masked_input)
                out = self.hidden(out)
                outputs.append(self.output(out).unsqueeze(0))
            return torch.cat(outputs, axis=0)
        out = self.input(x)
        out = self.hidden(out)
        return self.output(out)

