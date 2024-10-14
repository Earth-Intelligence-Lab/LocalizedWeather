import numpy as np
import torch

from Network import Helpers


class ERA5Network:
    def __init__(self, ERA5_data, MadisNetwork, n_neighbors_e2m):
        self.n_neighbors_e2m = n_neighbors_e2m
        self.MadisNetwork = MadisNetwork

        era5_lon_grid, era5_lat_grid = np.meshgrid(ERA5_data.longitude, ERA5_data.latitude)
        self.era5_pos = torch.Tensor(np.stack([era5_lon_grid, era5_lat_grid], axis=-1).reshape((-1, 2)))
        self.era5_lons = self.era5_pos[:, 0]
        self.era5_lats = self.era5_pos[:, 1]

        self.e2m_edge_index = Helpers.search_k_neighbors(self.MadisNetwork.pos, self.era5_pos, self.n_neighbors_e2m).long()
        self.e2m_relativeDistance = self.BuildIntepolationWeight(self.e2m_edge_index, self.era5_lons, self.era5_lats, self.MadisNetwork.stat_lons, self.MadisNetwork.stat_lats)


    def BuildIntepolationWeight(self, edges, lon_e, lat_e, lon_m, lat_m):

        lon_e = lon_e[edges[0,:]]
        lat_e = lat_e[edges[0, :]]
        lon_m = torch.from_numpy(lon_m[edges[1,:]], dtype=lon_e.dtype)
        lat_m = torch.from_numpy(lat_m[edges[1, :]], dtype=lat_e.dtype)


        delta_lon = lon_e - lon_m
        delta_lat = lat_e - lat_m
        delta_pos = (1./torch.sqrt(torch.square(delta_lon) + torch.square(delta_lat))).view(-1,1)

        labels = edges[1, :]
        labels_M = labels.view(labels.shape[0], 1).expand(-1, delta_pos.shape[0]).T

        M = torch.zeros(labels.max() + 1, len(delta_pos))
        M[labels_M, torch.arange(len(delta_pos))] = 1
        M = torch.mm(M, delta_pos)

        delta_pos = delta_pos / M[labels]

        return delta_pos