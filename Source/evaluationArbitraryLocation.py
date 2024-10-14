# This file contains functions to evaluate model performance.
# Author: Qidong Yang
# Date: 2024-02-14

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from Settings.Settings import StationSamplingMethod, StaticNodeType, ModelType


def evaluationArbitraryLocation(epoch,
                                model,
                                data_loader,
                                madis_network,
                                madis_norm_dict,
                                era5_norm_dict,
                                device,
                                lead_hrs,
                                loss_function = None,
                                optimizer=None,
                                save=False,
                                model_type=ModelType.GNN,
                                station_type='train',
                                show_progress_bar=False, figures_output_folder=None):

    is_train = station_type == 'train'

    MSE_u_sum = 0
    MSE_v_sum = 0

    MAE_u_sum = 0
    MAE_v_sum = 0

    if save == True:
        Pred_list = []
        Target_list = []

    if is_train:
        model.train()
    else:
        model.eval()

    print(f'runing {station_type}', flush=True)

    with torch.set_grad_enabled(is_train):
        loopItems = tqdm(data_loader) if show_progress_bar else data_loader
        for k, sample in enumerate(loopItems):
            madis_u = sample[f'madis_u'].to(device)
            madis_v = sample[f'madis_v'].to(device)
            madis_temp = sample[f'madis_temp'].to(device)
            # (n_batch, n_stations, n_times)


            madis_lon = sample[f'madis_lon'].to(device)
            madis_lat = sample[f'madis_lat'].to(device)

            # normalize input
            madis_u = madis_norm_dict['u'].encode(madis_u)
            madis_v = madis_norm_dict['v'].encode(madis_v)
            madis_temp = madis_norm_dict['temp'].encode(madis_temp)

            madis_matrix_len = madis_u.shape[2]

            x_madis_u = madis_u[:, :, :madis_matrix_len-lead_hrs]
            x_madis_v = madis_v[:, :, :madis_matrix_len-lead_hrs]
            x_madis_temp = madis_temp[:, :, :madis_matrix_len-lead_hrs]

            y_u = madis_u[:, :, [-1]]
            y_v = madis_v[:, :, [-1]]

            madis_x = torch.cat((x_madis_temp.unsqueeze(3), x_madis_u.unsqueeze(3), x_madis_v.unsqueeze(3)), dim=3)

            # b,s,ts,v = madis_x.shape
            #
            # madis_x = madis_x.view(b, s, -1)

            if era5_norm_dict is not None:
                era5_u = sample[f'era5_u'].to(device)
                era5_v = sample[f'era5_v'].to(device)
                era5_temp = sample[f'era5_temp'].to(device)
                # (n_batch, n_stations, n_times)

                era5_lon = sample[f'era5_lon'].to(device)
                era5_lat = sample[f'era5_lat'].to(device)

                era5_u = era5_norm_dict['u'].encode(era5_u)
                era5_v = era5_norm_dict['v'].encode(era5_v)
                era5_temp = era5_norm_dict['temp'].encode(era5_temp)

                edge_index_e2m = sample[f'e2m_edge_index'].to(device)

                era5_x = torch.cat((era5_temp.unsqueeze(3), era5_u.unsqueeze(3), era5_v.unsqueeze(3)), dim=3)

                b,s,t,v = era5_x.shape
                if model_type == ModelType.MLP:
                    era5_x = era5_x.view(b*s, t, v)
                else:
                    era5_x = era5_x.view(b, s, t*v)
            else:
                era5_lon = None
                era5_lat = None
                era5_x = None
                edge_index_e2m = None

            if is_train:
                optimizer.zero_grad()

            if model_type == ModelType.MLP:
                b,s,t,v = madis_x.shape
                madis_x = madis_x.view(b*s, t, v)
                out = model(madis_x, b, era5_x)
                _,v = out.shape
                out = out.view(b, s, v)
            else:
                out = model(madis_x,
                            madis_lon,
                            madis_lat,
                            era5_lon,
                            era5_lat,
                            era5_x,
                            edge_index_e2m)


            y_u = madis_norm_dict['u'].decode(y_u)
            y_v = madis_norm_dict['v'].decode(y_v)
            y = torch.cat((y_u, y_v), dim=2)

            # denormalize output
            out_u = madis_norm_dict['u'].decode(out[:, :, [0]])
            out_v = madis_norm_dict['v'].decode(out[:, :, [1]])
            out = torch.cat((out_u, out_v), dim=2)
            # (n_batch, n_stations, 2)


            if is_train:
                ls = loss_function(out, y)
                ls.backward()
                optimizer.step()


            if save == True:
                Pred_list.append(out.detach().cpu().numpy())
                Target_list.append(y.detach().cpu().numpy())

            y_u = y_u.detach()
            y_v = y_v.detach()

            out_u = out_u.detach()
            out_v = out_v.detach()

            mse_u = torch.sum(F.mse_loss(out_u, y_u, reduction='none'), dim=0).cpu().numpy()
            # (n_stations, 1)
            mse_v = torch.sum(F.mse_loss(out_v, y_v, reduction='none'), dim=0).cpu().numpy()
            # (n_stations, 1)

            MSE_u_sum = MSE_u_sum + np.sum(mse_u)
            MSE_v_sum = MSE_v_sum + np.sum(mse_v)


            mae_u = torch.sum(F.l1_loss(out_u, y_u, reduction='none'), dim=0).cpu().numpy()
            # (n_stations, 1)
            mae_v = torch.sum(F.l1_loss(out_v, y_v, reduction='none'), dim=0).cpu().numpy()
            # (n_stations, 1)

            MAE_u_sum = MAE_u_sum + np.sum(mae_u)
            MAE_v_sum = MAE_v_sum + np.sum(mae_v)
    
    if save == True:
        return MAE_u_sum, MSE_u_sum, MAE_v_sum, MSE_v_sum, np.concatenate(Pred_list, axis=0), np.concatenate(Target_list, axis=0)

    else:
        return MAE_u_sum, MSE_u_sum, MAE_v_sum, MSE_v_sum


def plot_metric(train_losses, valid_losses, test_losses, eval_interval, metric_name, output_path, y_range=None):

    fig, axs = plt.subplots(1, 1, figsize=(10, 5))

    epochs = len(train_losses)
    axs.plot(np.arange(1, epochs + 1), train_losses, label='Train')
    axs.plot(np.arange(1, epochs + 1), valid_losses, label='Valid')

    x_part_1 = np.array([1])
    x_part_2 = np.arange(1, len(test_losses)) * eval_interval
    x_axis = np.concatenate([x_part_1, x_part_2])

    axs.plot(x_axis, test_losses, label='Test')

    if y_range != None:
        axs.set_ylim(y_range[0], y_range[1])

    axs.legend()
    axs.grid()
    axs.set_xlabel('Epochs')
    axs.set_ylabel(metric_name)

    axs.set_title(metric_name + ' Plot')

    plt.savefig(os.path.join(output_path, '_'.join(metric_name.split(' ')) + '_plot.png'))
    plt.close()

