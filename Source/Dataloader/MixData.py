# Author: Qidong Yang & Jonathan Giezendanner

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dateutil import rrule
from torch.utils.data import Dataset

from Dataloader.Madis import Madis
from Normalization.Normalizers import MinMaxNormalizer


class MixData(Dataset):
    def __init__(self, year, back_hrs, lead_hours, meta_station, madis_network, madis_vars,
                 external_network, external_vars, external_data_object, data_path=Path('')):

        self.year = year
        self.back_hrs = back_hrs
        self.lead_hours = lead_hours
        self.madis_network = madis_network

        self.external_vars = external_vars
        self.madis_vars = madis_vars

        self.external_network = external_network
        self.external_data_object = external_data_object

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

        self.madis_mins_dict = dict()
        self.madis_maxs_dict = dict()
        self.madis_means_dict = dict()
        self.madis_stds_dict = dict()
        self.madis_ns_dict = dict()

        for madis_var in self.madis_vars:
            self.madis_mins_dict[madis_var] = np.min(self.madis_data[madis_var.name].values)
            self.madis_maxs_dict[madis_var] = np.max(self.madis_data[madis_var.name].values)
            self.madis_means_dict[madis_var] = np.mean(self.madis_data[madis_var.name].values)
            self.madis_stds_dict[madis_var] = np.std(self.madis_data[madis_var.name].values)
            self.madis_ns_dict[madis_var] = self.madis_data[madis_var.name].values.size

        self.madis_elv_min = np.min(self.madis_data.elv.values)
        self.madis_elv_max = np.max(self.madis_data.elv.values)

        if self.external_data_object is not None:
            self.external_mins_dict = dict()
            self.external_maxs_dict = dict()
            self.external_means_dict = dict()
            self.external_stds_dict = dict()
            self.external_ns_dict = dict()

            for external_var in self.external_vars:
                self.external_mins_dict[external_var] = np.min(self.external_data_object.data[external_var.name].values)
                self.external_maxs_dict[external_var] = np.max(self.external_data_object.data[external_var.name].values)
                self.external_means_dict[external_var] = np.mean(self.external_data_object.data[external_var.name].values)
                self.external_stds_dict[external_var] = np.std(self.external_data_object.data[external_var.name].values)
                self.external_ns_dict[external_var] = self.external_data_object.data[external_var.name].values.size

        if self.external_data_object is not None:
            self.lat_normalizer = MinMaxNormalizer(self.external_data_object.lat_low, self.external_data_object.lat_up)
            self.lon_normalizer = MinMaxNormalizer(self.external_data_object.lon_low, self.external_data_object.lon_up)
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

        sample = {
            f'time': torch.tensor([t_val.astype(np.int64) for t_val in time_sel.values]),
            f'madis_lon': self.lon_normalizer.encode(self.madis_network.madis_lon),
            f'madis_lat': self.lat_normalizer.encode(self.madis_network.madis_lat),
            f'k_edge_index': self.madis_network.k_edge_index,
        }

        madis_data = self.madis_data.sel(time=slice(time_sel[0], time_sel[-1]))
        for madis_var in self.madis_vars:
            madis_val = madis_data[madis_var.name].values.astype(np.float32)
            madis_val = torch.from_numpy(madis_val)
            sample[madis_var] = madis_val

            var_name_is_real = madis_var.name + '_is_real'
            madis_val_is_real = madis_data[var_name_is_real].values.astype(np.float32)
            sample[var_name_is_real] = madis_val_is_real


        if self.external_data_object is not None:
            if self.external_network is not None:
                sample[f'ex2m_edge_index'] = self.external_network.ex2m_edge_index
                sample[f'external_lon'] = self.lon_normalizer.encode(self.external_network.lons)
                sample[f'external_lat'] = self.lat_normalizer.encode(self.external_network.lats)
            else:
                sample[f'ex2m_edge_index'] = 1
                sample[f'external_lon'] = 1
                sample[f'external_lat'] = 1


            for external_var in self.external_vars:
                external_val = self.external_data_object.getSample(time_sel, external_var.name, self.external_network, self.back_hrs+1, self.lead_hours)
                sample['ext_' + external_var.name] = external_val

        return sample

