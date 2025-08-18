# Author: Qidong Yang & Jonathan Giezendanner

from pathlib import Path

import numpy as np
import torch
import xarray as xr

from Network.NetworkUtils import search_k_neighbors


class HRRR:
    def __init__(self, meta_station, madis_network, year, reanalysis_only=True, resampling_index_raw=None,
                 resampling_index=None,
                 region='Northeastern', data_path=Path('')):

        self.data_path = data_path
        self.resampling_index_raw = resampling_index_raw
        self.resampling_index = resampling_index
        self.year = year
        self.reanalysis_only = reanalysis_only
        self.region = region

        self.meta_output_folder = data_path / f'HRRR/{region}/Processed/Daily/Meta/'
        self.madis_output_folder = data_path / f'HRRR/{region}/Processed/Daily/Madis/{meta_station.filtered_file_name}/'

        self.data_file = self.data_path / f'HRRR/{region}/Processed/Yearly/Madis/{meta_station.filtered_file_name}/{year}.nc'

        self.data = self.get_hrrr(year, meta_station, madis_network, data_path, region)

        self.data = self.data.rename(dict(
            {'u': 'u80', 'v': 'v80',
             'u10': 'u', 'v10': 'v',
             't2m': 'temp',
             'd2m': 'dewpoint',
             'sdswrf': 'solar_radiation'}))

        self.n_nodes = self.data.sizes['node']
        self.data['node'] = np.arange(0, self.n_nodes)

        self.lat_low = np.min(self.data.latitude.values)
        self.lat_up = np.max(self.data.latitude.values)
        self.lon_low = np.min(self.data.longitude.values)
        self.lon_up = np.max(self.data.longitude.values)

    def LoadDataToMemory(self):
        self.data = self.data.load()

    def getSample(self, time_sel, variable, network, Madis_len, lead_time):
        if self.reanalysis_only:
            return torch.from_numpy(
                self.data[variable].isel(step=0).sel(time=slice(time_sel[0], time_sel[-1])).values.T.astype(np.float32))

        values = np.concatenate([
            self.data[variable].isel(step=0).sel(time=slice(time_sel[0], time_sel[Madis_len - 1])).values,  # historical
            self.data[variable].sel(time=time_sel[Madis_len - 1]).isel(step=slice(1, lead_time + 1)).values  # future
        ], axis=0).T.astype(np.float32)

        return torch.from_numpy(values)

    # region process_hrrr

    def get_hrrr(self, year, meta_station, madis_network, data_path, region):
        if self.data_file.exists():
            return xr.open_mfdataset(self.data_file)

        files = list(self.madis_output_folder.glob(f'{year}*.nc'))
        if len(files) >= 365:
            return self.create_yearly_hrrr()

        files = list(self.meta_output_folder.glob(f'{year}*.nc'))
        if len(files) >= 365:
            self.resampling_index = self.process_all_daily_meta_to_daily_madis(madis_network, year,
                                                                               self.resampling_index)
            return self.create_yearly_hrrr()

        self.resampling_index_raw = self.process_all_daily_raw_to_daily_meta(meta_station, year, data_path, region,
                                                                             self.resampling_index_raw)
        self.resampling_index = self.process_all_daily_meta_to_daily_madis(madis_network, year, self.resampling_index)
        return self.create_yearly_hrrr()

    def create_yearly_hrrr(self):
        self.data_file.parent.mkdir(exist_ok=True, parents=True)
        files = list(self.madis_output_folder.glob(f'{self.year}*.nc'))
        hrrr = xr.open_mfdataset(files, combine='by_coords').load()
        hrrr = hrrr.sortby(['time', 'step', 'node'])

        hrrr_vars = list(hrrr.data_vars.keys())
        for hrrr_var in hrrr_vars:
            hrrr[hrrr_var][:, 0, :] = hrrr[hrrr_var].isel(step=0).interpolate_na(dim='time').values
            hrrr[hrrr_var][:, 0, :] = hrrr[hrrr_var].isel(step=0).bfill(dim='time').values
            hrrr[hrrr_var][:, 0, :] = hrrr[hrrr_var].isel(step=0).ffill(dim='time').values
            hrrr[hrrr_var][:, 0, :] = hrrr[hrrr_var].isel(step=0).fillna(
                np.nanmean(hrrr[hrrr_var][:, 0, :].values.flatten())).values
            hrrr[hrrr_var].values = hrrr[hrrr_var].interpolate_na(dim='step').values
            hrrr[hrrr_var].values = hrrr[hrrr_var].bfill(dim='step').values
            hrrr[hrrr_var].values = hrrr[hrrr_var].ffill(dim='step').values

        hrrr.to_netcdf(self.data_file)
        return hrrr

    def process_all_daily_meta_to_daily_madis(self, madis_network, year, resampling_index=None):
        self.madis_output_folder.mkdir(exist_ok=True, parents=True)
        files = list(self.meta_output_folder.glob(f'{year}*.nc'))

        if resampling_index is None:
            file = files[0]
            hrrr = xr.open_mfdataset(file)
            pos = torch.Tensor(np.stack([hrrr.longitude, hrrr.latitude], axis=-1).reshape((-1, 2)))
            resampling_index = search_k_neighbors(madis_network.pos, pos, 9)[0, :].long().unique().numpy()

        for k, file in enumerate(files):
            print(f'file: {file}', flush=True)
            target_file = self.madis_output_folder / file.name
            if target_file.exists():
                continue

            hrrr = xr.open_mfdataset(file).load()
            hrrr = hrrr.sel(node=resampling_index).load()
            hrrr = hrrr.drop('original_index')
            hrrr.to_netcdf(target_file)

        return resampling_index

    def process_all_daily_raw_to_daily_meta(self, meta_station, year, data_path, region, resampling_index_raw=None):
        self.meta_output_folder.mkdir(exist_ok=True, parents=True)
        files = list(data_path.glob(f'HRRR/{region}/Raw/{year}*.nc'))

        if resampling_index_raw is None:
            processed_files = list(self.meta_output_folder.glob(f'*.nc'))
            if len(processed_files) > 0:
                resampling_index_raw = xr.open_mfdataset(processed_files[0]).original_index.values
            else:
                resampling_index_raw = self.get_raw_index(meta_station, files[0])

        for file in files:
            self.process_file_daily_raw_to_daily_meta(file, resampling_index_raw, self.meta_output_folder)

        return resampling_index_raw

    def process_file_daily_raw_to_daily_meta(self, file, neighbours, data_path):
        print(f'processing file {file.name}', flush=True)
        target_filename = data_path / file.name
        if target_filename.exists():
            return

        hrrr = xr.open_mfdataset(file)
        lon = hrrr.longitude.values.reshape(-1)
        lon[lon > 180] = lon[lon > 180] % 180 - 180
        lat = hrrr.latitude.values.reshape(-1)

        nb_times = len(hrrr.time)
        nb_steps = len(hrrr.step)

        hrrr_vars = list(hrrr.data_vars.keys())
        data_dic = dict()
        for hrrr_var in hrrr_vars:
            data_dic[hrrr_var] = (['time', 'step', 'node'],
                                  hrrr[hrrr_var].values.reshape((nb_times, nb_steps, -1))[:, :, neighbours])

        processed_data = xr.Dataset(
            data_vars=data_dic,
            coords={
                'node': np.arange(0, len(neighbours)),
                'step': hrrr.step.values,
                'time': hrrr.time.values,
                'original_index': (['node'], neighbours),
                'longitude': (['node'], lon[neighbours]),
                'latitude': (['node'], lat[neighbours])
            },
        )

        processed_data.to_netcdf(target_filename)

    def get_raw_index(self, meta_station, file):
        hrrr = xr.open_mfdataset(file)
        lon = hrrr.longitude.values.reshape(-1)
        lon[lon > 180] = lon[lon > 180] % 180 - 180
        lat = hrrr.latitude.values.reshape(-1)
        pos = torch.Tensor(np.stack([lon, lat], axis=-1).reshape((-1, 2)))
        station_index = meta_station.stations_raw.num.values / (meta_station.n_years * 366 * 24) > .5
        metaPos = torch.Tensor([[p.x, p.y] for p in meta_station.stations_raw.geometry.values[station_index]])
        neighbours = search_k_neighbors(metaPos, pos, 16)[0, :].long().unique().numpy()
        return neighbours
