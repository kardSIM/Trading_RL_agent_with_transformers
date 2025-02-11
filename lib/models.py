import numpy as np
import typing as tt
import torch
import torch.nn as nn


class SimpleFFDQN(nn.Module):
    def __init__(self, obs_len: int, actions_n: int):
        super(SimpleFFDQN, self).__init__()

        self.fc_val = nn.Sequential(
            nn.Linear(obs_len, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(obs_len, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        val = self.fc_val(x)
        adv = self.fc_adv(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))


class DQNConv1D(nn.Module):
    def __init__(self, shape: tt.Tuple[int, ...], actions_n: int):
        super(DQNConv1D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(shape[0], 128, 5),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5),
            nn.ReLU(),
            nn.Flatten(),
        )
        size = self.conv(torch.zeros(1, *shape)).size()[-1]

        self.fc_val = nn.Sequential(
            nn.Linear(size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(size, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(x)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + (adv - adv.mean(dim=1, keepdim=True))


class DQNConv1DLarge(nn.Module):
    def __init__(self, shape, actions_n):
        super(DQNConv1DLarge, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(shape[0], 32, 3),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.ReLU(),
            nn.Flatten(),
        )
        size = self.conv(torch.zeros(1, *shape)).size()[-1]

        self.fc_val = nn.Sequential(
            nn.Linear(size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(size, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(x)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + (adv - adv.mean(dim=1, keepdim=True))


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout=0.0, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens
        )
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, : X.shape[1], :].to(X.device)
        return self.dropout(X)




class DQNTransformer(nn.Module):
    def __init__(self, shape: tt.Tuple[int, ...],
                actions_n: int, 
                d_model=64, 
                n_heads=4, 
                num_layers=2,
                dropout: float = 0.1,
                ff_multiplier: int = 4, 
                activation: str = "relu"):
        

        super(DQNTransformer, self).__init__()

        self.input_projection = nn.Linear(shape[0], d_model, bias=False)

        self.pos_encoder = PositionalEncoding(d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            dim_feedforward=d_model*ff_multiplier,
            activation=activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, 
            num_layers=num_layers
        )

        self.fc_val = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, actions_n)
        )
        self._src_mask = None

    def _generate_square_subsequent_mask(self, sz, reset_mask=False):
        if self._src_mask is None or reset_mask:
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = (
                mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, float(0.0))
            )
            self._src_mask = mask
        return self._src_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  

        mask = self._generate_square_subsequent_mask(x.shape[1]).to(x.device)
        x = self.input_projection(x)

        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, mask)
        
        val = self.fc_val(x[:, -1, :])
        adv = self.fc_adv(x[:, -1, :])
        return val + (adv - adv.mean(dim=1, keepdim=True))

