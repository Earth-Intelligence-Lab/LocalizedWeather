import torch
from torch import nn as nn

from Modules.Activations import Tanh


class MPNN_MLP(nn.Module):
    def __init__(self, Madis_len, era5_len, n_out_features, hidden_dim=128):
        super(MPNN_MLP, self).__init__()

        self.Madis_len = Madis_len
        self.era5_len = era5_len
        self.hidden_dim = hidden_dim
        self.in_merge_dim = self.hidden_dim * 3
        self.n_out_features = n_out_features

        self.embedding_mlp = nn.Sequential(
            nn.Linear(self.Madis_len * 3 + 2, self.hidden_dim),
            Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            Tanh())

        self.output_mlp = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                        Tanh(),
                                        nn.Linear(self.hidden_dim, self.n_out_features))



        self.ex_embed_net_1 = nn.Sequential(nn.Linear(self.era5_len * 3 + 2, hidden_dim),
                                            Tanh()
                                            )

        self.ex_embed_net_2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                            Tanh()
                                            )

        self.message_net_1 = nn.Sequential(nn.Linear(hidden_dim + hidden_dim + 2, hidden_dim),
                                           Tanh()
                                           )
        self.message_net_2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                           Tanh()
                                           )

        self.update_net_1 = nn.Sequential(nn.Linear(hidden_dim + hidden_dim, hidden_dim),
                                          Tanh()
                                          )

        self.update_net_2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                          Tanh()
                                          )



        self.ex_embed_net_1_b = nn.Sequential(nn.Linear(self.era5_len * 3 + 2, hidden_dim),
                                            Tanh()
                                            )

        self.ex_embed_net_2_b = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                            Tanh()
                                            )

        self.message_net_1_b = nn.Sequential(nn.Linear(hidden_dim + hidden_dim + 2, hidden_dim),
                                           Tanh()
                                           )
        self.message_net_2_b = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                           Tanh()
                                           )

        self.update_net_1_b = nn.Sequential(nn.Linear(hidden_dim + hidden_dim, hidden_dim),
                                          Tanh()
                                          )

        self.update_net_2_b = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                          Tanh()
                                          )



    def forward(self, madis_x, pos, era5_x, era5_pos):

        x = torch.cat((madis_x, pos), dim=-1)

        x = self.embedding_mlp(x)

        era5_x_embedded = self.ex_embed_net_1(torch.cat((era5_x, era5_pos), dim=1))
        era5_x_embedded = self.ex_embed_net_2(era5_x_embedded)

        message = self.message_net_1(torch.cat((x, era5_x_embedded, era5_pos), dim=1))
        message = self.message_net_2(message)

        update = self.update_net_1(torch.cat((x, message), dim=-1))
        update = self.update_net_2(update)

        x = x + update

        era5_x_embedded = self.ex_embed_net_1_b(torch.cat((era5_x, era5_pos), dim=1))
        era5_x_embedded = self.ex_embed_net_2_b(era5_x_embedded)

        message = self.message_net_1_b(torch.cat((x, era5_x_embedded, era5_pos), dim=1))
        message = self.message_net_2_b(message)

        update = self.update_net_1_b(torch.cat((x, message), dim=-1))
        update = self.update_net_2_b(update)

        x = x + update

        x = self.output_mlp(x)

        return x
