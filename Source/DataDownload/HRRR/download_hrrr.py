import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from herbie import Herbie
from datetime import datetime, timedelta


def downloader_hourly(run_time, var_list, lat_min, lat_max, lon_min, lon_max, max_lead_hr=18):

    # run_time: time string like "2019-07-19 15:00"
    # var_list: variable list string like "(?:TMP:2 m|:UGRD:10 m)"

    datasets = []

    for lead_hr in range(max_lead_hr + 1):
        H = Herbie(run_time, model="hrrr", product="sfc", fxx=lead_hr, verbose=False)

        if H.grib == None:
            break
        else:
            dss = H.xarray(search=var_list)
            dss_new = []

            for ds in dss:
                ds = ds.drop_vars([coord for coord in ds.coords if coord not in ['latitude', 'longitude', 'time', 'step']])

                location_msk = (ds.latitude >= lat_min) & (ds.latitude <= lat_max) & (ds.longitude >= lon_min) & (ds.longitude <= lon_max)
                ds = ds.where(location_msk, drop=True)

                dss_new.append(ds)

            dataset = xr.merge(dss_new, compat='identical')

            datasets.append(dataset)

    Dataset = xr.concat(datasets, dim='step')

    return Dataset


def downloader_daily(date, var_list, lat_min, lat_max, lon_min, lon_max, max_lead_hr=18):

    # date: datetime object

    run_times = [(date + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M') for i in range(24)]
    datasets = []

    for run_time in run_times:

        dataset = downloader_hourly(run_time, var_list, lat_min, lat_max, lon_min, lon_max, max_lead_hr)
        datasets.append(dataset)

    Dataset = xr.concat(datasets, dim='time')

    return Dataset


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--saving_path', default='tmp', type=str)
    parser.add_argument('--var_list', default='(?:TMP:2 m|:UGRD:10 m)', type=str)
    parser.add_argument('--lon_min', default=-85 + 360, type=float)
    parser.add_argument('--lon_max', default=-61 + 360, type=float)
    parser.add_argument('--lat_min', default=33, type=float)
    parser.add_argument('--lat_max', default=52, type=float)
    parser.add_argument('--year', default=2019, type=int)
    parser.add_argument('--month', default=1, type=int)
    parser.add_argument('--day', default=1, type=int)

    args = parser.parse_args()
    date = datetime(args.year, args.month, args.day)

    print('Download Configuration', flush=True)
    print('var_list: ', args.var_list, flush=True)
    print('lon_min: ', args.lon_min, flush=True)
    print('lon_max: ', args.lon_max, flush=True)
    print('lat_min: ', args.lat_min, flush=True)
    print('lat_max: ', args.lat_max, flush=True)
    print('date: ', date, flush=True)
    print('saving_path: ', args.saving_path, flush=True)

    filename = args.saving_path + f'{args.year:04}{args.month:02}{args.day:02}.nc'
    Dataset = downloader_daily(date, args.var_list, args.lat_min, args.lat_max, args.lon_min, args.lon_max, max_lead_hr=18)
    Dataset.to_netcdf(filename)

    print(f'Saved to {filename}!')


if __name__ == '__main__':

    main()

