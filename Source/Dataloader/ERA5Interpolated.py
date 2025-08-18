# Author: Qidong Yang & Jonathan Giezendanner

from pathlib import Path

import numpy as np
import torch
import xarray as xr

from Dataloader.ERA5 import ERA5
from Network.NetworkUtils import search_k_neighbors
from Settings.Settings import InterpolationType


class ERA5Interpolated(ERA5):
    def __init__(self, year, madis_network, meta_station, n_neighbors_e2m, interpolation_type=InterpolationType.none,
                 region='World', zarr=False,
                 data_path=Path('')):
        self.interpolation_type = interpolation_type
        self.filtered_file_name = meta_station.filtered_file_name
        ERA5.__init__(self, year, madis_network, meta_station, n_neighbors_e2m, region, zarr, data_path)

    def get_ERA5(self, region, madis_network, n_neighbors_e2m):
        self.interpolated_file_path = self.GetInterpolatedFilePath(n_neighbors_e2m)
        if self.interpolated_file_path.exists():
            return xr.open_mfdataset(self.interpolated_file_path, engine=self.engine).load()

        self.data = super().get_ERA5(region, madis_network, n_neighbors_e2m,
                                     use_node_data=self.interpolation_type in [InterpolationType.Stacked])
        return self.make_interpolated()

    def GetInterpolatedFilePath(self, n_neighbors_e2m):
        return self.data_path / 'ERA5' / 'Interpolated' / f'era5interpolated_e2m_{n_neighbors_e2m}_{self.interpolation_type.name}_{self.year}_{self.filtered_file_name}{self.extension}'

    def make_interpolated(self):
        if self.interpolated_file_path.exists():
            return self.data

        if self.interpolation_type == InterpolationType.Stacked:
            self.data = self.data[['u', 'v', 'temp', 'dewpoint']].sortby('node')
            self.data = self.data.drop(['expver', 'number'])
            pos = torch.from_numpy(np.stack([self.data.longitude, self.data.latitude], axis=-1).astype(np.float32))
            era5nodes = search_k_neighbors(self.madis_network.pos, pos, self.n_neighbors_e2m).long().numpy()
            # self.data['node'] = np.arange(len(self.data.node))
            self.data = self.data.isel(node=era5nodes[0, :])
            self.data['node'] = np.tile(np.arange(self.n_neighbors_e2m), len(self.madis_network.pos))
            self.data = self.data.assign_coords(stations=("node", era5nodes[1, :] + 1))
            self.data = self.data.set_index(sample=("stations", "node")).unstack('sample')
            self.data = self.data.transpose('stations', ...)

            self.SaveInterpolatedFile(self.data)
            return self.data

        if self.interpolation_type == InterpolationType.Nearest:
            self.data = self.data[['u', 'v', 'temp', 'dewpoint']].sortby(['latitude', 'longitude'])
            self.data = self.data.drop(['expver', 'number'])
            self.data = self.data.interp(latitude=('stations', self.madis_network.pos[:, 1]),
                                         longitude=('stations', self.madis_network.pos[:, 0]), method='nearest',
                                         assume_sorted=True)[['u', 'v', 'temp', 'dewpoint']].load()

            self.data['stations'] = self.data.stations.values + 1
            self.data = self.data.transpose('stations', ...)

            self.SaveInterpolatedFile(self.data)
            return self.data

        if self.interpolation_type == InterpolationType.BiCubic:
            self.data = self.data[['u', 'v', 'temp', 'dewpoint']].sortby(['latitude', 'longitude'])
            self.data = self.data.drop(['expver', 'number'])
            self.data = self.data.interp(latitude=('stations', self.madis_network.pos[:, 1]),
                                         longitude=('stations', self.madis_network.pos[:, 0]), method='cubic',
                                         assume_sorted=True)[['u', 'v', 'temp', 'dewpoint']].load()

            self.data['stations'] = self.data.stations.values + 1
            self.data = self.data.transpose('stations', ...)

            self.SaveInterpolatedFile(self.data)
            return self.data

    def getSample(self, time_sel, variable, network, Madis_len, lead_time):
        val = torch.from_numpy(self.data[variable].sel(time=slice(time_sel[0], time_sel[-1])).values.astype(np.float32))
        if self.interpolation_type == InterpolationType.Stacked:
            val = val.view(val.size(0), -1)
        return val

    def RenameLatLon(self, data):
        return data.rename(dict(
            {
                'lon': 'longitude',
                'lat': 'latitude'
            }))

    def SaveInterpolatedFile(self, data):
        self.interpolated_file_path.parent.mkdir(exist_ok=True, parents=True)

        data.to_netcdf(self.interpolated_file_path)
