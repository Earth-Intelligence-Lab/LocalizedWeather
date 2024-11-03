# Author: Qidong Yang & Jonathan Giezendanner

import torch
from torch import nn as nn
from torch_geometric.nn import MessagePassing, InstanceNorm

from Modules.Activations import Tanh


class GNN_Layer_External(MessagePassing):
    """
    External message passing layer
    """

    def __init__(self, in_dim, out_dim, hidden_dim, ex_in_dim):
        """
        Initialize message passing layers
        """
        super(GNN_Layer_External, self).__init__(node_dim=-2, aggr='mean')

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.ex_in_dim = ex_in_dim

        self.ex_embed_net_1 = nn.Sequential(nn.Linear(ex_in_dim + 2, hidden_dim),
                                            Tanh()
                                            )
        self.ex_embed_net_2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                            Tanh()
                                            )
        self.message_net_1 = nn.Sequential(nn.Linear(in_dim + hidden_dim + 2, hidden_dim),
                                           Tanh()
                                           )
        self.message_net_2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                           Tanh()
                                           )
        self.update_net_1 = nn.Sequential(nn.Linear(in_dim + hidden_dim, hidden_dim),
                                          Tanh()
                                          )
        self.update_net_2 = nn.Sequential(nn.Linear(hidden_dim, out_dim),
                                          Tanh()
                                          )
        self.norm = InstanceNorm(out_dim)

    def forward(self, in_x, ex_x, in_pos, ex_pos, edge_index, batch):
        """
        Propagate messages along edges
        """
        n_in_x = in_x.size(0)
        # n_batch * n_stations

        ex_x = self.ex_embed_net_1(torch.cat((ex_x, ex_pos), dim=1))
        ex_x = self.ex_embed_net_2(ex_x)

        x = torch.cat((in_x, ex_x), dim=0)
        pos = torch.cat((in_pos, ex_pos), dim=0)

        index_shift = torch.zeros_like(edge_index)
        index_shift[0] = index_shift[0] + n_in_x

        x = self.propagate(edge_index + index_shift, x=x, pos=pos)
        x = x[:n_in_x]
        x = self.norm(x, batch)
        return x

    def message(self, x_i, x_j, pos_i, pos_j):
        """
        Message update following formula 8 of the paper
        """
        message = self.message_net_1(torch.cat((x_i, x_j, pos_i - pos_j), dim=-1))
        message = self.message_net_2(message)
        return message

    def update(self, message, x):
        """
        Node update following formula 9 of the paper
        """
        update = self.update_net_1(torch.cat((x, message), dim=-1))
        update = self.update_net_2(update)
        return x + update
