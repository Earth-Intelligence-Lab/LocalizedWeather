# Author: Qidong Yang & Jonathan Giezendanner

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely import points


class Madis(object):
    def __init__(self, times, coords_raw, coords, lat_low, lat_up, lon_low, lon_up, file_name, filtered_file_name,
                 n_years=5,
                 root_path=Path('')):

        # n_years comes from meta station

        self.times = times
        self.coords_raw = coords_raw  # list of geometry points
        self.coords = coords  # list of geometry points
        self.n_years = n_years
        self.years = self.times.dt.year.data
        self.months = self.times.dt.month.data
        self.days = self.times.dt.day.data
        self.hours = self.times.dt.hour.data

        self.lat_low = lat_low
        self.lat_up = lat_up
        self.lon_low = lon_low
        self.lon_up = lon_up

        self.lons = np.array([i.x for i in self.coords])
        self.lats = np.array([i.y for i in self.coords])

        self.lons_raw = np.array([i.x for i in self.coords_raw])
        self.lats_raw = np.array([i.y for i in self.coords_raw])

        self.root_path = root_path

        meta_year_cover = f'Meta--{2024 - self.n_years}--2023'
        meta_year_folder = self.root_path / f'madis/processed/{meta_year_cover}'
        # meta_year_folder = self.root_path/f'madis/SyntheticERA5/{meta_year_cover}'
        madis_raw_filename = f'madis_{self.years[0]}_{file_name}.nc'
        madis_filename = f'madis_{self.years[0]}_{filtered_file_name}.nc'

        if os.path.exists(meta_year_folder) == False:
            os.system(f'mkdir -p {meta_year_folder}')

        self.madis_raw_ds_path = f'{meta_year_folder}/{madis_raw_filename}'
        self.madis_ds_path = f'{meta_year_folder}/{madis_filename}'

        if os.path.exists(self.madis_ds_path):
            self.ds_xr = xr.open_dataset(self.madis_ds_path)
        else:
            rawdata = self.createRawFile()
            self.ds_xr = self.createFile(rawdata)

    def createRawFile(self):
        if (os.path.exists(self.madis_raw_ds_path)):
            rawData = xr.open_dataset(self.madis_raw_ds_path)
        else:

            madis_ds = self.loadData()

            madis_ds = np.stack(madis_ds, axis=-1)
            # (n_stations, n_variables, n_times)

            n_variables = madis_ds.shape[1]
            madis_var_is_real = np.zeros(madis_ds.shape, dtype=bool)

            for i in range(n_variables):
                madis_var = madis_ds[:, i, :]

                # keep track of nans
                madis_var_is_real[:, i, :] = ~np.isnan(madis_var)

                # fill NaN
                madis_var = pd.DataFrame(madis_var.T).ffill().bfill().values.T
                madis_var = np.nan_to_num(madis_var, nan=np.nanmean(madis_var))
                madis_ds[:, i, :] = madis_var

            rawData = xr.Dataset(
                {
                    'u': (['stations', 'time'], madis_ds[:, 0, :]),
                    'u_is_real': (['stations', 'time'], madis_var_is_real[:, 0, :]),
                    'v': (['stations', 'time'], madis_ds[:, 1, :]),
                    'v_is_real': (['stations', 'time'], madis_var_is_real[:, 1, :]),
                    'temp': (['stations', 'time'], madis_ds[:, 2, :]),
                    'temp_is_real': (['stations', 'time'], madis_var_is_real[:, 2, :]),
                    'dewpoint': (['stations', 'time'], madis_ds[:, 3, :]),
                    'dewpoint_is_real': (['stations', 'time'], madis_var_is_real[:, 3, :]),
                    'solar_radiation': (['stations', 'time'], madis_ds[:, 4, :]),
                    'solar_radiation_is_real': (['stations', 'time'], madis_var_is_real[:, 4, :]),
                    'elv': (['stations'], madis_ds[:, 5, 0]),
                    'lon': (['stations'], self.lons_raw),
                    'lat': (['stations'], self.lats_raw),
                },
                coords={
                    'stations': np.arange(1, len(self.coords_raw) + 1),
                    'time': self.times.values,
                },
            )

            rawData.to_netcdf(self.madis_raw_ds_path)
        return rawData

    def loadData(self):
        madis_ds = []
        for year in np.unique(self.years):
            print("processing madis year", year, flush=True)
            year_ind = self.years == year
            for month in np.unique(self.months[year_ind]):
                print("processing madis month", month, flush=True)
                data = self.load_madis_monthly(year, month)
                ind = (self.months == month) & year_ind
                day_hour = zip(self.days[ind], self.hours[ind])
                for day, hour in day_hour:
                    da = self.load_madis_hourly(data, year, month, day, hour)
                    madis_ds.append(da)
        return madis_ds

    def load_madis_monthly(self, year, month):
        data_path = self.root_path / f'madis/raw_monthly/mesonet/{year}/{month}.nc'

        data = xr.open_dataset(data_path)

        wind_speed_check = ((data.windSpeedDD == b'S') + (data.windSpeedDD == b'V'))
        wind_direction_check = ((data.windDirDD == b'S') + (data.windDirDD == b'V'))
        temperature_check = ((data.temperatureDD == b'S') + (data.temperatureDD == b'V'))
        # Todo this should account for dynamic wind speed
        wind_speed_amplitude_check = (data.windSpeed < 50)
        validobs = wind_speed_check & wind_direction_check & temperature_check & wind_speed_amplitude_check

        data = data.sel(index=validobs)

        lat_obs = data.latitude.values
        lon_obs = data.longitude.values

        ind = (self.lat_low <= lat_obs) & (lat_obs <= self.lat_up)
        ind = ind & ((self.lon_low <= lon_obs) & (lon_obs <= self.lon_up))

        data = data.sel(index=ind)

        return data

    def load_madis_hourly(self, data, year, month, day, hour):
        data = data.sel(index=((data.year == year) & (data.month == month) & (data.day == day) & (data.hour == hour)))

        if len(data.index) > 0:
            lat_obs = data.latitude.values
            lon_obs = data.longitude.values
            elv_obs = data.elevation.values
            ws_obs = data.windSpeed.values
            wd_obs = data.windDir.values
            temp_obs = data.temperature.values
            dewpoint_obs = data.dewpoint.values
            solarRadiation_obs = data.solarRadiation.values

            dewpoint_check = ((data.dewpoint.values == b'S') + (data.dewpoint.values == b'V'))
            dewpoint_obs[~dewpoint_check] = np.nan
            solarRadiation_check = ((data.solarRadiation.values == b'S') + (data.solarRadiation.values == b'V'))
            solarRadiation_obs[~solarRadiation_check] = np.nan

            u_obs = np.cos(np.deg2rad(270 - wd_obs)) * ws_obs
            v_obs = np.sin(np.deg2rad(270 - wd_obs)) * ws_obs

            coords = points(np.concatenate([lon_obs.reshape(-1, 1), lat_obs.reshape(-1, 1)], axis=1))

            df_agg = (gpd.GeoDataFrame(
                {
                    'u': u_obs,
                    'v': v_obs,
                    'temp': temp_obs,
                    'elv': elv_obs,
                    'dewpoint': dewpoint_obs,
                    'solar_radiation': solarRadiation_obs
                }, geometry=coords).groupby('geometry', as_index=False).mean())
            df_agg = gpd.GeoDataFrame(df_agg)

            da = np.ones((len(self.coords_raw), 6)) * np.nan
            query_idx = df_agg.sindex.query(self.coords_raw, predicate='contains')
            da[query_idx[0, :], 0] = df_agg.u.values[query_idx[1, :]]
            da[query_idx[0, :], 1] = df_agg.v.values[query_idx[1, :]]
            da[query_idx[0, :], 2] = df_agg.temp.values[query_idx[1, :]]
            da[query_idx[0, :], 3] = df_agg.dewpoint.values[query_idx[1, :]]
            da[query_idx[0, :], 4] = df_agg.solar_radiation.values[query_idx[1, :]]
            da[query_idx[0, :], 5] = df_agg.elv.values[query_idx[1, :]]

        else:
            da = np.ones((len(self.coords_raw), 6)) * np.nan

        return da

    def createFile(self, rawdata):
        ind = [e in self.coords for e in self.coords_raw]
        xrdata = rawdata.sel(stations=ind)
        xrdata['stations'] = np.arange(1, len(self.coords) + 1)
        xrdata.to_netcdf(self.madis_ds_path)
        return xrdata
