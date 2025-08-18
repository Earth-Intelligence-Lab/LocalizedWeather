# Author: Qidong Yang & Jonathan Giezendanner

from typing import Tuple, Optional

import torch
from torch import nn as nn

from Modules.Transformer.StationsEmbedding import StationEmbed
from Modules.Transformer.Transformer import Transformer


class VisionTransformer(nn.Module):
    def __init__(self, n_stations, madis_len: int, madis_n_vars_i: int, madis_n_vars_o: int, dim: int, attn_dim: int,
                 mlp_dim: int, num_heads: int, num_layers: int, era5_n_vars=None, era5_len=None):
        # n_stations       the number of stations
        # madis_len        the length of each madis variable
        # madis_n_vars_i   the number of madis variable as input
        # madis_n_vars_o   the number of madis variable as output
        # dim              embedding dimension
        # attn_dim         the hidden dimension of the attention layer
        # mlp_dim          the hidden layer dimension of the FFN
        # num_heads        the number of heads in the attention layer
        # num_layers       the number of attention layers
        # era5_len         the length of each era5 variable
        # era5_n_vars      the number of era5 variable as input

        super().__init__()

        self.station_embed = StationEmbed(madis_len=madis_len, madis_n_vars_i=madis_n_vars_i, era5_len=era5_len,
                                          era5_n_vars=era5_n_vars, hidden_dim=dim)
        self.pos_E = nn.Embedding(n_stations, dim)  # positional embedding matrix

        self.transformer = Transformer(dim=dim, attn_dim=attn_dim, mlp_dim=mlp_dim, num_heads=num_heads,
                                       num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(dim, madis_n_vars_o)
        )

    def forward(self, madis_x: torch.Tensor, era5_x=None, return_attn=False) -> Tuple[
        torch.Tensor, Optional[torch.Tensor]]:
        # madis_x      (n_batch, n_stations, madis_len, nb_var)
        # era5_x       (n_batch, n_stations, era5_len, nb_var)
        # return_attn  whether to return the attention alphas
        #
        # Outputs
        # out          the output of the vision transformer. shape: (B, nout)
        # alphas       the attention weights for all heads and layers. None if return_attn is False, otherwise
        #              shape: (B, num_layers, num_heads, n_stations, n_stations)

        # generate embeddings
        embs = self.station_embed(madis_x, era5_x)  # station embedding
        # (n_batch, n_stations, dim)

        B, T, _ = embs.shape
        pos_ids = torch.arange(T).expand(B, -1).to(embs.device)
        embs += self.pos_E(pos_ids)  # positional embedding
        # (n_batch, n_stations, dim)

        x = embs
        # (n_batch, n_stations, dim)

        x, alphas = self.transformer(x, attn_mask=None, return_attn=return_attn)
        # (n_batch, n_stations, dim)

        out = self.head(x)
        # (n_batch, n_stations, madis_n_vars_o)

        return out, alphas
