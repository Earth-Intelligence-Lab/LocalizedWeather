# Author: Qidong Yang & Jonathan Giezendanner
from pathlib import Path

import numpy as np
import xarray as xr


class ERA5(object):
    def __init__(self, lat_low, lat_up, lon_low, lon_up, year, region='Northeastern',
                 root_path=Path('')):

        buffer = 1.5

        self.lat_low = np.floor(lat_low - buffer)
        self.lat_up = np.ceil(lat_up + buffer)
        self.lon_low = np.floor(lon_low - buffer)
        self.lon_up = np.ceil(lon_up + buffer)
        self.year = year
        self.root_path = root_path

        self.file_path = self.root_path / 'ERA5' / 'Processed' / f'era5_{year}_{self.lon_low:.0f}_{self.lon_up:.0f}_{self.lat_low:.0f}_{self.lat_up:.0f}.nc'

        self.data = self.load_ERA5(region)

        self.u_min = self.data.u10.values.min()
        self.u_max = self.data.u10.values.max()
        self.v_min = self.data.v10.values.min()
        self.v_max = self.data.v10.values.max()
        self.t_min = self.data.t2m.values.min()
        self.t_max = self.data.t2m.values.max()
        # self.d_min = data.d2m.values.min()
        # self.d_max = data.d2m.values.max()

    def load_ERA5(self, region):

        if self.file_path.exists():
            return xr.open_dataset(self.file_path)

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

        self.file_path.parent.mkdir(exist_ok=True, parents=True)
        data.to_netcdf(self.file_path)

        return data

    def load_ERA5_monthly(self, month, region):

        # data_path = self.root_path/f'ERA5/World/surface_{self.year}_{month}.nc'
        data_path = self.root_path / 'ERA5' / region / f'surface_{self.year}_{month}.nc'

        data = xr.open_dataset(data_path)
        data = data.sel(longitude=slice(self.lon_low, self.lon_up), latitude=slice(self.lat_up, self.lat_low))
        data = data.rename({'valid_time': 'time'})

        return data
