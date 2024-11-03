# Author: Qidong Yang & Jonathan Giezendanner

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dateutil import rrule
from torch.utils.data import Dataset

from Source.Dataloader.ERA5 import ERA5
from Source.Dataloader.Madis import Madis
from Source.Normalization.Normalizers import MinMaxNormalizer


class MixData(Dataset):
    def __init__(self, year, back_hrs, lead_hours, meta_station, madis_network, n_neighbors_m2m, era5_network,
                 data_path=Path('')):
        # meta_station: MetaStation object

        self.year = year
        self.back_hrs = back_hrs
        self.lead_hours = lead_hours
        self.n_neighbors_m2m = n_neighbors_m2m
        self.madis_network = madis_network

        self.era5_network = era5_network
        if self.era5_network is not None:
            self.ERA5 = ERA5(meta_station.lat_low, meta_station.lat_up, meta_station.lon_low, meta_station.lon_up,
                             self.year, data_path=data_path)
            self.era5_data = self.ERA5.data

        self.time_line = pd.to_datetime(pd.Series(list(
            rrule.rrule(rrule.HOURLY, dtstart=datetime.strptime(f'{year}-01-01', '%Y-%m-%d'),
                        until=datetime.strptime(f'{year + 1}-01-01', '%Y-%m-%d')))[0:-1])).to_xarray()

        self.stations = meta_station.stations
        stations_raw = meta_station.stations_raw
        self.stat_coords = list(self.stations['geometry'])
        stat_coords_raw = list(stations_raw['geometry'])
        self.stat_lons = np.array([i.x for i in self.stat_coords])
        self.stat_lats = np.array([i.y for i in self.stat_coords])
        self.n_stations = len(self.stat_coords)

        self.Madis = Madis(self.time_line, stat_coords_raw, self.stat_coords, meta_station.lat_low, meta_station.lat_up,
                           meta_station.lon_low, meta_station.lon_up, meta_station.file_name,
                           meta_station.filtered_file_name, meta_station.n_years, data_path=data_path)
        self.madis_data = self.Madis.ds_xr

        self.madis_u_min = np.min(self.madis_data.u.values)
        self.madis_u_max = np.max(self.madis_data.u.values)

        self.madis_v_min = np.min(self.madis_data.v.values)
        self.madis_v_max = np.max(self.madis_data.v.values)

        self.madis_temp_min = np.min(self.madis_data.temp.values)
        self.madis_temp_max = np.max(self.madis_data.temp.values)

        self.madis_elv_min = np.min(self.madis_data.elv.values)
        self.madis_elv_max = np.max(self.madis_data.elv.values)

        if self.era5_network is not None:
            self.era5_u_min = self.ERA5.u_min
            self.era5_u_max = self.ERA5.u_max

            self.era5_v_min = self.ERA5.v_min
            self.era5_v_max = self.ERA5.v_max

            self.era5_temp_min = self.ERA5.t_min
            self.era5_temp_max = self.ERA5.t_max

        if self.era5_network is not None:
            self.lat_normalizer = MinMaxNormalizer(self.ERA5.lat_low, self.ERA5.lat_up)
            self.lon_normalizer = MinMaxNormalizer(self.ERA5.lon_low, self.ERA5.lon_up)
        else:
            self.lat_normalizer = MinMaxNormalizer(self.Madis.lat_low, self.Madis.lat_up)
            self.lon_normalizer = MinMaxNormalizer(self.Madis.lon_low, self.Madis.lon_up)

    def __len__(self):

        return len(self.time_line) - self.back_hrs - self.lead_hours

    def __getitem__(self, index):
        # index_end: the step to predict

        index_start = index
        index_end = index + self.back_hrs + self.lead_hours

        time_sel = self.time_line[index_start:index_end + 1]

        madis_u = self.madis_data.u.sel(time=slice(time_sel[0], time_sel[-1])).values.astype(np.float32)
        madis_v = self.madis_data.v.sel(time=slice(time_sel[0], time_sel[-1])).values.astype(np.float32)
        madis_temp = self.madis_data.temp.sel(time=slice(time_sel[0], time_sel[-1])).values.astype(np.float32)
        # (n_stations, n_times)

        madis_u = torch.from_numpy(madis_u)
        madis_v = torch.from_numpy(madis_v)
        madis_temp = torch.from_numpy(madis_temp)

        sample = {
            f'madis_u': madis_u,
            f'madis_v': madis_v,
            f'madis_temp': madis_temp,
            f'madis_lon': self.lon_normalizer.encode(self.madis_network.madis_lon),
            f'madis_lat': self.lat_normalizer.encode(self.madis_network.madis_lat),
            f'k_edge_index': self.madis_network.k_edge_index,
        }

        if self.era5_network is not None:
            era5_u = torch.from_numpy(
                np.moveaxis(self.era5_data.u10.sel(time=slice(time_sel[0], time_sel[-1])).values, 0, -1).reshape(
                    (self.era5_network.era5_pos.size(0), -1)).astype(np.float32))
            era5_v = torch.from_numpy(
                np.moveaxis(self.era5_data.v10.sel(time=slice(time_sel[0], time_sel[-1])).values, 0, -1).reshape(
                    (self.era5_network.era5_pos.size(0), -1)).astype(np.float32))
            era5_temp = torch.from_numpy(
                np.moveaxis(self.era5_data.t2m.sel(time=slice(time_sel[0], time_sel[-1])).values, 0, -1).reshape(
                    (self.era5_network.era5_pos.size(0), -1)).astype(np.float32))

            sample[f'e2m_edge_index'] = self.era5_network.e2m_edge_index
            sample[f'era5_u'] = era5_u
            sample[f'era5_v'] = era5_v
            sample[f'era5_temp'] = era5_temp
            sample[f'era5_lon'] = self.lon_normalizer.encode(self.era5_network.era5_lons)
            sample[f'era5_lat'] = self.lat_normalizer.encode(self.era5_network.era5_lats)

        return sample
