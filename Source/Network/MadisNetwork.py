import numpy as np
import torch
from torch_geometric.nn import knn_graph


class MadisNetwork:
    def __init__(self, meta_station, n_neighbors_m2m):
        # meta_station: MetaStation object

        self.n_neighbors_m2m = n_neighbors_m2m

        self.stations = meta_station.stations
        self.stat_coords = list(self.stations['geometry'])
        self.stat_lons = np.array([i.x for i in self.stat_coords])
        self.stat_lats = np.array([i.y for i in self.stat_coords])
        self.n_stations = len(self.stat_coords)

        self.stations = np.arange(self.n_stations)

        self.pos = torch.cat(
            [torch.from_numpy(self.stat_lons.reshape((-1, 1))), torch.from_numpy(self.stat_lats.reshape((-1, 1)))],
            dim=1)

        self.madis_lon = torch.from_numpy(self.stat_lons).reshape((-1, 1))
        self.madis_lat = torch.from_numpy(self.stat_lats).reshape((-1, 1))

        self.k_edge_index = self.BuildMadisNetwork(self.madis_lon, self.madis_lat)

    def BuildMadisNetwork(self, lon, lat):
        pos = torch.cat([lon, lat], dim=1)
        k_edge_index = knn_graph(pos, k=self.n_neighbors_m2m, batch=torch.zeros((len(pos),)), loop=False)

        return k_edge_index
