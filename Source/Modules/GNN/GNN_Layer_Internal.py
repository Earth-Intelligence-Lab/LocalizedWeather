import torch
from torch import nn
from torch_geometric.nn import MessagePassing, InstanceNorm

from Source.Modules.Activations import Tanh


class GNN_Layer_Internal(MessagePassing):
    """
    Internal message passing layer
    """

    def __init__(self, in_dim, out_dim, hidden_dim, org_in_dim):
        """
        Initialize message passing layers
        """
        super(GNN_Layer_Internal, self).__init__(node_dim=-2, aggr='mean')

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.message_net_1 = nn.Sequential(nn.Linear(2 * in_dim + org_in_dim + 2, hidden_dim),
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

    def forward(self, x, u, pos, edge_index, batch):
        """
        Propagate messages along edges
        """
        x = self.propagate(edge_index, x=x, u=u, pos=pos)
        x = self.norm(x, batch)
        return x

    def message(self, x_i, x_j, u_i, u_j, pos_i, pos_j):
        """
        Message update following formula 8 of the paper
        """
        message = self.message_net_1(torch.cat((x_i, x_j, u_i - u_j, pos_i - pos_j), dim=-1))
        message = self.message_net_2(message)
        return message

    def update(self, message, x):
        """
        Node update following formula 9 of the paper
        """
        update = self.update_net_1(torch.cat((x, message), dim=-1))
        update = self.update_net_2(update)
        return x + update
