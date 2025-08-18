# Author: Qidong Yang & Jonathan Giezendanner

import numpy as np
import torch

from Network.NetworkUtils import search_k_neighbors


class HRRNetwork:
    def __init__(self, HRRR_data, MadisNetwork, n_neighbors_h2m):
        self.n_neighbors_h2m = n_neighbors_h2m
        self.MadisNetwork = MadisNetwork

        self.pos = torch.Tensor(np.stack([HRRR_data.longitude, HRRR_data.latitude], axis=-1).reshape((-1, 2)))
        self.lons = self.pos[:, 0]
        self.lats = self.pos[:, 1]

        self.ex2m_edge_index = search_k_neighbors(self.MadisNetwork.pos, self.pos, self.n_neighbors_h2m).long()
