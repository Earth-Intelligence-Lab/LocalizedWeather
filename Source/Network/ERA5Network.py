import numpy as np
import torch


class ERA5Network:
    def __init__(self, ERA5_data, MadisNetwork, n_neighbors_e2m):
        self.n_neighbors_e2m = n_neighbors_e2m
        self.MadisNetwork = MadisNetwork

        era5_lon_grid, era5_lat_grid = np.meshgrid(ERA5_data.longitude, ERA5_data.latitude)
        self.era5_pos = torch.Tensor(np.stack([era5_lon_grid, era5_lat_grid], axis=-1).reshape((-1, 2)))
        self.era5_lons = self.era5_pos[:, 0]
        self.era5_lats = self.era5_pos[:, 1]

        self.e2m_edge_index = self.search_k_neighbors(self.MadisNetwork.pos, self.era5_pos, self.n_neighbors_e2m).long()
        self.e2m_relativeDistance = self.BuildIntepolationWeight(self.e2m_edge_index, self.era5_lons, self.era5_lats,
                                                                 self.MadisNetwork.stat_lons,
                                                                 self.MadisNetwork.stat_lats)

    def search_k_neighbors(self, base_points, cand_points, k):
        # base_points: (n_b, n_features)
        # cand_points: (n_c, n_features)

        dis = torch.sum((base_points.unsqueeze(1) - cand_points.unsqueeze(0)) ** 2, dim=-1)
        _, inds = torch.topk(dis, k, dim=1, largest=False)

        n_b = base_points.size(0)

        j_inds = inds.view((1, -1))
        i_inds = (torch.arange(n_b).view((-1, 1)) * torch.ones((n_b, k))).view((1, -1))

        edge_index = torch.cat([j_inds, i_inds], dim=0)

        return edge_index

    def BuildIntepolationWeight(self, edges, lon_e, lat_e, lon_m, lat_m):
        lon_e = lon_e[edges[0, :]]
        lat_e = lat_e[edges[0, :]]
        lon_m = torch.from_numpy(lon_m[edges[1, :]].astype(np.float32))
        lat_m = torch.from_numpy(lat_m[edges[1, :]].astype(np.float32))

        delta_lon = lon_e - lon_m
        delta_lat = lat_e - lat_m
        delta_pos = (1. / torch.sqrt(torch.square(delta_lon) + torch.square(delta_lat))).view(-1, 1)

        labels = edges[1, :]
        labels_M = labels.view(labels.shape[0], 1).expand(-1, delta_pos.shape[0]).T

        M = torch.zeros(labels.max() + 1, len(delta_pos))
        M[labels_M, torch.arange(len(delta_pos))] = 1
        M = torch.mm(M, delta_pos)

        delta_pos = delta_pos / M[labels]

        return delta_pos
