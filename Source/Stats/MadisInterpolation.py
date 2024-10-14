import os
import numpy as np
import pandas as pd


def calculate_madis_interpolation_error(madis_network, Data_List, output_saving_path):
    u_mses = []
    v_mses = []
    u_maes = []
    v_maes = []
    for dataset in Data_List:

        madis_connections_known_unknown = madis_network.k_edge_index['test'][:, madis_network.k_edge_index_type_logical_known_unknown['test'][:, 0]]
        original_index_stations_known = madis_network.stations_original_index['test'][madis_connections_known_unknown[0, :].long()]
        original_index_stations_unknown = madis_network.stations_original_index['test'][madis_connections_known_unknown[1, :].long()]
        u_interpolated = dataset.madis_data.u.values[original_index_stations_known, :]
        v_interpolated = dataset.madis_data.v.values[original_index_stations_known, :]

        edge_weight = madis_network.edge_weights['test'][:, 0].numpy()
        edge_weight = np.array([edge_weight] * u_interpolated.shape[1]).T

        u_interpolated = pd.DataFrame(u_interpolated * edge_weight)
        u_interpolated['unknow_index'] = original_index_stations_unknown
        u_interpolated = u_interpolated.groupby('unknow_index').sum().reset_index()

        u = dataset.madis_data.u.values[u_interpolated.unknow_index, :]
        u_interpolated = u_interpolated.drop(columns=['unknow_index']).values

        v_interpolated = pd.DataFrame(v_interpolated * edge_weight)
        v_interpolated['unknow_index'] = original_index_stations_unknown
        v_interpolated = v_interpolated.groupby('unknow_index').sum().reset_index()

        v = dataset.madis_data.v.values[v_interpolated.unknow_index, :]
        v_interpolated = v_interpolated.drop(columns=['unknow_index']).values

        u_error = u - u_interpolated
        v_error = v - v_interpolated

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

    np.save(os.path.join(output_saving_path, f'test_madis_interpolated_u_mse.npy'), u_mse)
    np.save(os.path.join(output_saving_path, f'test_madis_interpolated_v_mse.npy'), v_mse)
    np.save(os.path.join(output_saving_path, f'test_madis_interpolated_u_mae.npy'), u_mae)
    np.save(os.path.join(output_saving_path, f'test_madis_interpolated_v_mae.npy'), v_mae)
    np.save(os.path.join(output_saving_path, f'test_madis_interpolated_mse.npy'), mse)
    np.save(os.path.join(output_saving_path, f'test_madis_interpolated_mae.npy'), mae)