import torch
from torch import nn as nn


class PlainMLP(nn.Module):
    def __init__(self, Madis_len, era5_len=None, hidden_dim=128):
        super(PlainMLP, self).__init__()

        self.Madis_len = Madis_len
        self.hidden_dim = hidden_dim
        self.in_merge_dim = self.hidden_dim * 3

        self.linear_u = nn.Sequential(nn.Linear(self.Madis_len, self.hidden_dim), nn.ReLU(inplace=True))
        self.linear_v = nn.Sequential(nn.Linear(self.Madis_len, self.hidden_dim), nn.ReLU(inplace=True))
        self.linear_temp = nn.Sequential(nn.Linear(self.Madis_len, self.hidden_dim), nn.ReLU(inplace=True))

        self.era5_len = era5_len
        if self.era5_len is not None:
            self.in_merge_dim += self.hidden_dim * 3

            self.era5_linear_u = nn.Sequential(nn.Linear(self.era5_len, self.hidden_dim), nn.ReLU(inplace=True))
            self.era5_linear_v = nn.Sequential(nn.Linear(self.era5_len, self.hidden_dim), nn.ReLU(inplace=True))
            self.era5_linear_temp = nn.Sequential(nn.Linear(self.era5_len, self.hidden_dim), nn.ReLU(inplace=True))

        self.merge_net = nn.Sequential(
            nn.Linear(self.in_merge_dim, self.in_merge_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_merge_dim // 2, self.in_merge_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_merge_dim // 4, 2)
        )

    def forward(self, madis_x, batch, era5_x=None):

        # madis_u: (n_batch * n_stations, Madis_len)
        # madis_v: (n_batch * n_stations, Madis_len)
        # madis_temp: (n_batch * n_stations, Madis_len)

        madis_u_emb = self.linear_u(madis_x[:, :, 0])
        madis_v_emb = self.linear_v(madis_x[:, :, 1])
        madis_temp_emb = self.linear_temp(madis_x[:, :, 2])
        # (n_batch, n_stations, hidden_dim)

        all_emb = torch.cat((madis_u_emb, madis_v_emb, madis_temp_emb), dim=1)
        # (n_batch, n_stations, in_merge_dim)

        if self.era5_len is not None:
            era5_u_emb = self.era5_linear_u(era5_x[:, :, 0])
            era5_v_emb = self.era5_linear_v(era5_x[:, :, 1])
            era5_temp_emb = self.era5_linear_temp(era5_x[:, :, 2])
            all_emb = torch.cat((all_emb, era5_u_emb, era5_v_emb, era5_temp_emb), dim=1)

        out = self.merge_net(all_emb)
        # (n_batch, n_stations, 2)

        return out
