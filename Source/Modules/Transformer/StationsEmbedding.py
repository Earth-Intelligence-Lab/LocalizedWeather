# Author: Qidong Yang & Jonathan Giezendanner

import torch
from torch import nn

class StationEmbed(nn.Module):
    def __init__(self, madis_len: int, madis_n_vars_i: int, era5_len=None, era5_n_vars=None, hidden_dim=128):
        # madis_len       the length of each madis variable
        # madis_n_vars_i  the number of madis variable as input
        # era5_len        the length of each era5 variable
        # era5_n_vars     the number of era5 variable as input

        super().__init__()

        self.madis_len = madis_len
        self.hidden_dim = hidden_dim
        self.in_merge_dim = self.hidden_dim * madis_n_vars_i

        self.madis_n_vars_i = madis_n_vars_i
        self.era5_n_vars = era5_n_vars
        self.era5_len = era5_len

        self.encoding_madis_layers = nn.ModuleList(modules=(
            nn.Sequential(nn.Linear(self.madis_len, self.hidden_dim), nn.ReLU(inplace=True))
            for _ in range(self.madis_n_vars_i)))

        if self.era5_len is not None:
            self.in_merge_dim += self.hidden_dim * self.era5_n_vars

            self.encoding_era5_layers = nn.ModuleList(modules=(
                nn.Sequential(nn.Linear(self.era5_len, self.hidden_dim), nn.ReLU(inplace=True))
                for _ in range(self.era5_n_vars)))

        self.merge_net = nn.Sequential(nn.Linear(self.in_merge_dim, self.hidden_dim))


    def forward(self, madis_x: torch.Tensor, era5_x=None):
        # madis_x        (n_batch, n_stations, madis_len, nb_var)
        # era5_x         (n_batch, n_stations, era5_len, nb_var)
        #
        # Output
        # out      the patch embeddings for the input. shape: (n_batch, n_stations, hidden_dim)

        all_emb = torch.cat([self.encoding_madis_layers[i](madis_x[:, :, :, i]) for i in range(self.madis_n_vars_i)], dim=-1)
        # (n_batch, n_stations, hidden_dim * madis_n_vars_i)

        if self.era5_len is not None:
            era5_emb = torch.cat([self.encoding_era5_layers[i](era5_x[:, :, :, i]) for i in range(self.era5_n_vars)], dim=-1)
            # (n_batch, n_stations, hidden_dim * era5_n_vars)
            all_emb = torch.cat((all_emb, era5_emb), dim=-1)
            # (n_batch, n_stations, hidden_dim * era5_n_vars + hidden_dim * madis_n_vars_i)

        out = self.merge_net(all_emb)
        # (n_batch, n_stations, hidden_dim)

        return out