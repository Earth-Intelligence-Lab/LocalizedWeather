import os
import numpy as np
import pandas as pd


def calculate_ERA5_interpolation_error(era5_network, madis_network, Data_List, output_saving_path):
    u_mses = []
    v_mses = []
    u_maes = []
    v_maes = []
    for dataset in Data_List:

        madis_stations_unknown_orig_index = madis_network.ind_orig_stations_logical_unknown['test']
        madis_data = dataset.madis_data
        madis_u = madis_data.u.values[madis_stations_unknown_orig_index]
        madis_v = madis_data.v.values[madis_stations_unknown_orig_index]

        data = dataset.era5_data
        u10 = np.moveaxis(data.u10.values, 0, -1).reshape((era5_network.era5_pos.size(0), -1))
        u10_sub = u10[era5_network.e2m_edge_index[0, :]]
        v10 = np.moveaxis(data.v10.values, 0, -1).reshape((era5_network.era5_pos.size(0), -1))
        v10_sub = v10[era5_network.e2m_edge_index[0, :]]

        rel_distances_era5 = era5_network.e2m_relativeDistance.cpu().numpy()

        df = pd.DataFrame(columns=['u', 'v'])

        for k in range(u10_sub.shape[0]):
            df.loc[k, 'u'] = u10_sub[k, :] * rel_distances_era5[k].squeeze()
            df.loc[k, 'v'] = v10_sub[k, :] * rel_distances_era5[k].squeeze()

        df['madis_index'] = era5_network.e2m_edge_index[1, :]

        sumdf = df.groupby('madis_index').sum()

        u = np.stack(sumdf.u.values)[madis_stations_unknown_orig_index]
        v = np.stack(sumdf.v.values)[madis_stations_unknown_orig_index]

        u_error = u - madis_u
        v_error = v - madis_v

        u_mse = u_error ** 2
        v_mse = v_error ** 2

        u_mae = np.abs(u_error)
        v_mae = np.abs(v_error)

        u_mses.append(u_mse)
        v_mses.append(v_mse)

        u_maes.append(u_mae)
        v_maes.append(v_mae)

    u_mses = np.concatenate(u_mses, axis=1)
    v_mses = np.concatenate(v_mses, axis=1)

    u_maes = np.concatenate(u_maes, axis=1)
    v_maes = np.concatenate(v_maes, axis=1)

    u_mse = np.mean(u_mses)
    v_mse = np.mean(v_mses)

    u_mae = np.mean(u_maes)
    v_mae = np.mean(v_maes)

    mse = np.mean(u_mses + v_mses)
    mae = np.mean(u_maes + v_maes)

    np.save(os.path.join(output_saving_path, f'test_era5vsmadis_interpolated_u_mse.npy'), u_mse)
    np.save(os.path.join(output_saving_path, f'test_era5vsmadis_interpolated_v_mse.npy'), v_mse)
    np.save(os.path.join(output_saving_path, f'test_era5vsmadis_interpolated_u_mae.npy'), u_mae)
    np.save(os.path.join(output_saving_path, f'test_era5vsmadis_interpolated_v_mae.npy'), v_mae)
    np.save(os.path.join(output_saving_path, f'test_era5vsmadis_interpolated_mse.npy'), mse)
    np.save(os.path.join(output_saving_path, f'test_era5vsmadis_interpolated_mae.npy'), mae)