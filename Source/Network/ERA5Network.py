# Author: Qidong Yang & Jonathan Giezendanner

import numpy as np
import torch
from Network.NetworkUtils import search_k_neighbors


class ERA5Network:
    def __init__(self, ERA5_data, MadisNetwork, n_neighbors_e2m):
        self.n_neighbors_e2m = n_neighbors_e2m
        self.MadisNetwork = MadisNetwork

        # era5_lon_grid, era5_lat_grid = np.meshgrid(ERA5_data.longitude, ERA5_data.latitude)
        self.lons = torch.from_numpy(ERA5_data.longitude.values.astype(np.float32))
        self.lats = torch.from_numpy(ERA5_data.latitude.values.astype(np.float32))
        self.pos = torch.Tensor(np.stack([self.lons, self.lats], axis=-1))

        self.GetNetwork()

    def GetNetwork(self):
        self.ex2m_edge_index = search_k_neighbors(self.MadisNetwork.pos, self.pos, self.n_neighbors_e2m).long()

