# Author: Qidong Yang & Jonathan Giezendanner
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from Network.NetworkUtils import search_k_neighbors


class ERA5(object):
    def __init__(self, year, madis_network, meta_station, n_neighbors_e2m, region='World', zarr=False,
                 data_path=Path('')):

        self.madis_network = madis_network

        lat_low = meta_station.lat_low
        lat_up = meta_station.lat_up
        lon_low = meta_station.lon_low
        lon_up = meta_station.lon_up

        buffer = 1.5

        self.extension = '.nc'
        self.engine = 'netcdf4'

        self.lat_low = np.floor(lat_low - buffer)
        self.lat_up = np.ceil(lat_up + buffer)
        self.lon_low = np.floor(lon_low - buffer)
        self.lon_up = np.ceil(lon_up + buffer)
        self.year = year
        self.data_path = data_path
        self.n_neighbors_e2m = n_neighbors_e2m

        self.node_file_path = self.data_path / f'ERA5/Processed/era5_{year}_e2m_{n_neighbors_e2m}_{meta_station.filtered_file_name}{self.extension}'
        self.region_file_path = self.data_path / 'ERA5' / 'Processed' / (
                f'era5_{self.year}_{self.lon_low:.0f}_{self.lon_up:.0f}_{self.lat_low:.0f}_{self.lat_up:.0f}' + self.extension)

        self.data = self.get_ERA5(region, madis_network, n_neighbors_e2m)

    def LoadDataToMemory(self):
        self.data = self.data.load()

    def get_ERA5(self, region, madis_network, n_neighbors_e2m, use_node_data=True):
        self.use_node_data = use_node_data
        if use_node_data and self.node_file_path.exists():
            return self.RenameVariables(xr.open_mfdataset(self.node_file_path, engine=self.engine))
        # else:
        #     print(f'File ERA5 fie {self.node_file_path} does not exist')
        #     sys.exit()

        region_data = self.load_era5_region(region, self.region_file_path)
        if not use_node_data:
            return self.RenameVariables(region_data)

        return self.RenameVariables(self.build_nodes_data(region_data, madis_network, n_neighbors_e2m))

    def build_nodes_data(self, region_data, madis_network, n_neighbors_e2m):
        region_data = region_data.stack(node=('longitude', 'latitude'))
        region_data = region_data.assign_coords(node_id=(['node'], range(len(region_data.node)))) \
            .set_index(node='node_id')
        pos = torch.from_numpy(np.stack([region_data.longitude, region_data.latitude], axis=-1).astype(np.float32))
        era5nodes = search_k_neighbors(madis_network.pos, pos, n_neighbors_e2m).long().numpy()[0, :]
        era5nodes = np.sort(np.unique(era5nodes))
        region_data = region_data.isel(node=era5nodes)
        region_data = region_data.chunk()
        region_data.to_netcdf(self.node_file_path)
        return region_data

    def load_era5_region(self, region, region_file_path):

        if region_file_path.exists():
            return xr.open_mfdataset(region_file_path, engine=self.engine)

        data_list = []

        for month in range(1, 13):
            data = self.load_ERA5_monthly(month, region)

            data_list.append(data)

        data = xr.concat(data_list, dim='time')
        data = data[[
            'u10',
            'v10',
            't2m',
            'd2m',
            'ssr'
        ]]

        region_file_path.parent.mkdir(exist_ok=True, parents=True)
        data.to_netcdf(region_file_path)

        return data

    def getSample(self, time_sel, variable, network, Madis_len, lead_time):
        val = torch.from_numpy(
            np.moveaxis(self.data[variable].sel(time=slice(time_sel[0], time_sel[-1])).values.astype(np.float32), 0,
                        -1))

        return val

    def load_ERA5_monthly(self, month, region):

        data_path = self.data_path / 'ERA5' / region / (f'surface_{self.year}_{month}' + self.extension)
        data = xr.open_mfdataset(data_path, engine=self.engine)

        data = data.sel(longitude=slice(self.lon_low, self.lon_up), latitude=slice(self.lat_up, self.lat_low))
        data = data.rename({'valid_time': 'time'})

        return data

    def RenameVariables(self, data):
        return data.rename(dict(
            {'u10': 'u', 'v10': 'v',
             't2m': 'temp',
             'd2m': 'dewpoint',
             'ssr': 'solar_radiation'}))
