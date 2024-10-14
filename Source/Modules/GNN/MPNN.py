import torch
from torch import nn as nn
from torch_geometric.data import Data

from Source.Modules.GNN.GNN_Layer_External import GNN_Layer_External
from Source.Modules.GNN.GNN_Layer_Internal import GNN_Layer_Internal
from Source.Modules.Activations import Tanh


class MPNN(nn.Module):
    def __init__(self,
                 n_passing,
                 lead_hrs,
                 n_node_features_m,
                 n_node_features_e,
                 n_out_features,
                 hidden_dim=128):

        super(MPNN, self).__init__()

        self.lead_hrs = lead_hrs
        self.n_node_features_m = n_node_features_m
        self.n_node_features_e = n_node_features_e
        self.n_passing = n_passing
        self.hidden_dim = hidden_dim
        self.n_out_features = n_out_features

        self.gnn_ex_1 = GNN_Layer_External(in_dim=self.hidden_dim, out_dim=self.hidden_dim, hidden_dim=self.hidden_dim, ex_in_dim=self.n_node_features_e)
        self.gnn_ex_2 = GNN_Layer_External(in_dim=self.hidden_dim, out_dim=self.hidden_dim, hidden_dim=self.hidden_dim, ex_in_dim=self.n_node_features_e)


        self.gnn_layers = nn.ModuleList(modules=(
            GNN_Layer_Internal(
            in_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            org_in_dim=self.n_node_features_m)
            for _ in range(self.n_passing)))

        self.embedding_mlp = nn.Sequential(
            nn.Linear(self.n_node_features_m + 2, self.hidden_dim),
            Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            Tanh())


        self.output_mlp = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                        Tanh(),
                                        nn.Linear(self.hidden_dim, self.n_out_features))



    def build_graph_internal(self, x, madis_lon, madis_lat, edge_index):
        # madis_u: (n_batch, n_stations, n_times)
        # madis_v: (n_batch, n_stations, n_times)
        # madis_temp: (n_batch, n_stations, n_times)
        # madis_lon: (n_batch, n_stations, 1)
        # madis_lat: (n_batch, n_stations, 1)
        # edge_index: (n_batch, 2, n_edges)
        # edge_attr: (n_batch, n_edges, n_edge_features)

        n_batch = x.size(0)
        n_stations = x.size(1)
        n_edges = edge_index.size(2)

        # x = torch.cat((madis_temp, madis_u, madis_v), dim=2)
        # (n_batch, n_stations, n_times * 3)
        x = x.view(n_batch * n_stations, -1)
        # (n_batch * n_stations, n_times * 3)

        pos = torch.cat((madis_lon, madis_lat), dim=2)
        # (n_batch, n_stations, 2)
        pos = pos.view(n_batch * n_stations, -1)
        # (n_batch * n_stations, 2)

        # node_index = torch.arange(n_stations, dtype=torch.int64).view(1, -1) * torch.ones(n_batch, 1, dtype=torch.int64)
        # # (n_batch, n_stations, 2)
        # node_index = node_index.view(n_batch * n_stations, ).to(x.device)
        # # (n_batch * n_stations, 2)

        batch = torch.arange(n_batch).view(-1, 1) * torch.ones(1, n_stations)
        # (n_batch, n_stations)
        batch = batch.view(n_batch * n_stations, ).to(x.device)
        # (n_batch * n_stations, )




        index_shift = (torch.arange(n_batch) * n_stations).view(-1, 1, 1).to(x.device)
        edge_index = torch.cat(list(edge_index + index_shift), dim=1)
        # (2, n_batch * n_edges)

        graph = Data(x=x, pos=pos, batch=batch.long(), edge_index=edge_index.long())

        return graph

    def build_graph_external(self, madis_x, ex_x, ex_lon, ex_lat, edge_index):
        # madis_x: (n_batch, n_stations_m, n_features_m)
        # madis_lon: (n_batch, n_stations_m, 1)
        # madis_lat: (n_batch, n_stations_m, 1)
        # ex_x: (n_batch, n_stations_e, n_features_e)
        # ex_lon: (n_batch, n_stations_e, 1)
        # ex_lat: (n_batch, n_stations_e, 1)
        # edge_index: (n_batch, 2, n_edges)

        n_batch = madis_x.size(0)
        n_stations_m = madis_x.size(1)
        n_stations_e = ex_x.size(1)
        n_edges = edge_index.size(-1)
        '''
        madis_x = madis_x.view(n_batch * n_stations_m, -1)
        # (n_batch * n_stations_m, n_features_m)

        madis_pos = torch.cat((madis_lon, madis_lat), dim=2)
        # (n_batch, n_stations_m, 2)
        madis_pos = madis_pos.view(n_batch * n_stations_m, -1)
        # (n_batch * n_stations_m, 2)
        '''
        ex_x = ex_x.view(n_batch * n_stations_e, -1)
        # (n_batch * n_stations_e, n_features_e)

        ex_pos = torch.cat((ex_lon.view(n_batch, n_stations_e, 1), ex_lat.view(n_batch, n_stations_e, 1)), dim=2)
        # (n_batch, n_stations_e, 2)
        ex_pos = ex_pos.view(n_batch * n_stations_e, -1)
        # (n_batch * n_stations_e, 2)

        madis_shift = (torch.arange(n_batch) * n_stations_m).view((n_batch, 1))
        ex_shift = (torch.arange(n_batch) * n_stations_e).view((n_batch, 1))
        shift = torch.cat((ex_shift, madis_shift), dim=1).unsqueeze(-1).to(madis_x.device)
        edge_index = torch.cat(list(edge_index + shift), dim=1)
        # (2, n_batch * n_edges)
        '''
        batch = torch.arange(n_batch).view(-1, 1) * torch.ones(1, n_stations_m)
        # (n_batch, n_stations_m)
        batch = batch.view(n_batch * n_stations_m, ).to(madis_x.device)
        # (n_batch * n_stations_m, )
        '''
        graph = Data(x=ex_x, pos=ex_pos, edge_index=edge_index.long())

        return graph

    def forward(self,
                madis_x,
                madis_lon,
                madis_lat,
                edge_index,
                ex_lon,
                ex_lat,
                ex_x,
                edge_index_e2m):

        # madis_x: (n_batch, n_stations_m, n_hours_m, n_features_m)
        # madis_lon: (n_batch, n_stations_m, 1)
        # madis_lat: (n_batch, n_stations_m, 1)
        # edge_index: (n_batch, 2, n_edges_m2m)
        # edge_attr: (n_batch, n_edges_m2m, n_edge_features)
        # edge_index_e2m: (n_batch, 2, n_edges_e2m)

        # mask = (~torch.isin(edge_index[0, 0, :], arbitrary_nodes))
        # edge_index = edge_index[:, :, mask]
        # edge_attr = edge_attr[:, mask, :]

        n_batch, n_stations_m, n_hours_m, n_features_m = madis_x.shape
        # n_batch = madis_x.size(0)
        # n_stations_m = madis_x.size(1)
        #n_edges = edge_attr.size(1)

        madis_x = madis_x.view(n_batch, n_stations_m, -1)


        in_graph = self.build_graph_internalbuild_graph_internal(madis_x, madis_lon, madis_lat, edge_index)

        u = in_graph.x
        in_pos = in_graph.pos
        batch = in_graph.batch
        edge_index_m2m = in_graph.edge_index
        # 2, n_batch * n_stations * n_neighbours

        # arbitrary_nodes_mask = torch.isin(node_index, arbitrary_nodes)


        in_x = self.embedding_mlp(torch.cat((u, in_pos), dim=-1))



        if ex_x is not None:
            ex_graph = self.build_graph_external(madis_x, ex_x, ex_lon, ex_lat, edge_index_e2m)
            ex_x = ex_graph.x
            ex_pos = ex_graph.pos
            edge_index_e2m = ex_graph.edge_index


        if ex_x is not None:
            in_x = self.gnn_ex_1(in_x, ex_x, in_pos, ex_pos, edge_index_e2m, batch)

        for i in range(self.n_passing):
            in_x = self.gnn_layers[i](in_x, u, in_pos, edge_index_m2m, batch)
        # (n_batch * n_stations, hidden_dim)

        if ex_x is not None:
            in_x = self.gnn_ex_2(in_x, ex_x, in_pos, ex_pos, edge_index_e2m, batch)

        out = self.output_mlp(in_x)
        # (n_batch * n_stations, 2)
        out = out.view(n_batch, n_stations_m, self.n_out_features)
        # (n_batch, n_stations, 2)


        return out
