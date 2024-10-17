# Author: Qidong Yang & Jonathan Giezendanner

import os
from collections import Counter
from pathlib import Path

import geopandas as gpd
import pandas as pd
import xarray as xr
from shapely import points


class MetaStation(object):
    def __init__(self, lat_low, lat_up, lon_low, lon_up, n_years=5, control_ratio=0.9, shapefile_path=None,
                 root_path=Path('')):

        self.lat_low = lat_low
        self.lat_up = lat_up
        self.lon_low = lon_low
        self.lon_up = lon_up
        self.n_years = n_years
        self.control_ratio = control_ratio
        self.start_year = 2023 - n_years + 1
        self.root_path = root_path
        self.file_name = '%.2f_%.2f_%.2f_%.2f' % (self.lon_low, self.lon_up, self.lat_low, self.lat_up)
        self.shapefile_path = shapefile_path
        if self.shapefile_path is not None:
            self.file_name += self.shapefile_path.stem
        self.filtered_file_name = self.file_name + f'_filtered_{self.control_ratio}'

        self.all_station_file = root_path / (f'madis/stations/stations_{self.start_year}_2023_{self.file_name}.shp')
        self.station_file = root_path / (
            f'madis/stations/stations_{self.start_year}_2023_{self.filtered_file_name}.shp')
        self.data_dir = lambda year, month: root_path / f'madis/raw_monthly/mesonet/{year}/{month}.nc'

        Path(self.all_station_file).parent.mkdir(exist_ok=True, parents=True)

        if os.path.exists(self.all_station_file):
            self.stations_raw = gpd.read_file(self.all_station_file)
        else:
            self.stations_raw = self.generate_station_table()

        if os.path.exists(self.station_file):
            self.stations = gpd.read_file(self.station_file)
        elif os.path.exists(self.all_station_file):
            self.stations = self.generate_filtered_station_table(self.stations_raw)

    def generate_station_table(self):
        print('Generating station table')
        counter = Counter([])

        for year in range(self.start_year, 2024):
            print(f'Generating station table for year {year}', flush=True)
            for month in range(1, 13):
                print(f'Generating station table for month {month}', flush=True)

                data_dir = self.data_dir(year, month)
                data = xr.open_dataset(data_dir)

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

                if len(data.index) > 0:
                    coords_sub = \
                    data[['longitude', 'latitude', 'year', 'month', 'day', 'hour']].to_pandas().reset_index(
                        drop=True).drop_duplicates()[['longitude', 'latitude']].values
                    counter = counter + Counter(points(coords_sub))

        counter = gpd.GeoDataFrame(pd.Series(counter).reset_index().rename(columns={'index': 'geometry', 0: 'num'}))

        counter = counter.set_crs(epsg=4326)

        if self.shapefile_path is not None:
            roi = gpd.read_file(self.shapefile_path).dissolve()
            counter = counter[counter.geometry.within(roi.iloc[0].geometry)]

        counter.to_file(self.all_station_file, index=False)

        return counter

    def generate_filtered_station_table(self, counter):

        counter = counter[counter.num >= self.n_years * 366 * 24 * self.control_ratio]

        counter.to_file(self.station_file, index=False)

        return counter
