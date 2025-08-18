# Author: Qidong Yang & Jonathan Giezendanner

from pathlib import Path

import xarray as xr

from Dataloader.HRRR import HRRR
from Settings.Settings import InterpolationType


class HRRRInterpolated(HRRR):
    def __init__(self, meta_station, madis_network, year, reanalysis_only=True, resampling_index_raw=None,
                 resampling_index=None,
                 region='Northeastern', data_path=Path('')):
        HRRR.__init__(self, meta_station, madis_network, year, reanalysis_only, resampling_index_raw, resampling_index,
                      region, data_path)

    def make_interpolated(self, external_network, interpolation_type, madis_data, n_neighbors_ex2m, meta_station):
        interpolated_file_path = self.data_path / 'HRRR' / self.region / 'Interpolated' / f'HRRRinterpolated_h2m_{n_neighbors_ex2m}_{interpolation_type.name}_{self.year}_{meta_station.filtered_file_name}.nc'
        print(interpolated_file_path)
        if interpolated_file_path.exists():
            self.data = xr.open_mfdataset(interpolated_file_path)
            return

        print('creating file')

        if interpolation_type == InterpolationType.Nearest:
            if n_neighbors_ex2m != 1:
                raise ValueError("Nearest interpolation only supports n_neighbors_ex2m=1")

            self.data = self.data.sel(node=external_network.ex2m_edge_index[0, :]).load()

            interpolated_file_path.parent.mkdir(exist_ok=True, parents=True)
            self.data.to_netcdf(interpolated_file_path)
            return
        else:
            raise NotImplementedError
