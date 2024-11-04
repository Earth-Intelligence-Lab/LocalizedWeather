# This file is used to train wind correction model and evaluate it.
# Author: Qidong Yang & Jonathan Giezendanner
# Date: 2024-02-14

import os
import pickle
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset

import Normalization.NormalizerBuilder as NormalizerBuilder
from Dataloader.ERA5 import ERA5
from Dataloader.MetaStation import MetaStation
from Dataloader.MixData import MixData
from Dataloader.MixDataMLP import MixDataMLP
from EvaluateModel import evaluate_model
from Modules.GNN.MPNN import MPNN
from Modules.MLP.PlainMLP import PlainMLP
from Network.ERA5Network import ERA5Network
from Network.MadisNetwork import MadisNetwork
from Settings.Settings import ModelType
from Modules.MLP.MPNN_MLP import MPNN_MLP


def Run(args):
    data_path = Path(args.data_path)
    output_saving_path = data_path / args.output_saving_path
    output_saving_path.mkdir(exist_ok=True, parents=True)
    show_progress_bar = args.show_progress_bar
    shapefile_path = args.shapefile_path

    if shapefile_path is None:
        lon_low, lon_up, lat_low, lat_up = args.coords
    else:
        shapefile_path = data_path / shapefile_path
        lon_low, lat_low, lon_up, lat_up = gpd.read_file(shapefile_path).bounds.iloc[0].values

    back_hrs = args.back_hrs
    lead_hrs = args.lead_hrs
    whole_len = back_hrs + 1

    Madis_len = whole_len
    ERA5_len = whole_len + lead_hrs
    hidden_dim = args.hidden_dim
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    eval_interval = args.eval_interval
    weight_decay = args.weight_decay
    model_type = args.model_type
    madis_control_ratio = args.madis_control_ratio
    n_years = args.n_years
    n_passing = args.n_passing
    n_neighbors_m2m = args.n_neighbors_m2m

    n_neighbors_e2m = args.n_neighbors_e2m

    figures_path = output_saving_path / 'figures'
    figures_path.mkdir(exist_ok=True, parents=True)

    print('Experiment Configuration', flush=True)

    for k, v in vars(args).items():
        print(f'{k}: {v}', flush=True)

    ##### Set Random Seed #####
    np.random.seed(100)

    ##### Get Device #####
    device = 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    ##### Load Data #####
    meta_station = MetaStation(lat_low, lat_up, lon_low, lon_up, n_years, madis_control_ratio,
                               shapefile_path=shapefile_path, data_path=data_path)

    madis_network = MadisNetwork(meta_station, n_neighbors_m2m)

    if n_neighbors_e2m > 0:
        era5_stations = ERA5(meta_station.lat_low, meta_station.lat_up, meta_station.lon_low, meta_station.lon_up, 2023,
                             region='Northeastern',
                             data_path=data_path).data

        era5_network = ERA5Network(era5_stations, madis_network, n_neighbors_e2m)
    else:
        era5_network = None

    years = list(range(2024 - n_years, 2024))

    # for dataset in ['train', 'val', 'test']:
    if model_type == ModelType.GNN:
        Data_List = [MixData(year, back_hrs, lead_hrs, meta_station, madis_network, n_neighbors_m2m, era5_network,
                             data_path=data_path) for year in years]
    else:
        Data_List = [MixDataMLP(year, back_hrs, lead_hrs, meta_station, madis_network, n_neighbors_m2m, era5_network,
                                data_path=data_path) for year in years]

        if era5_network is not None:
            era5_network.era5_pos = torch.Tensor(Data_List[0].madis_data[['lon', 'lat']].to_dataarray().values.T)
            era5_network.era5_lons = torch.Tensor(Data_List[0].madis_data[['lon']].to_dataarray().values.T)
            era5_network.era5_lats = torch.Tensor(Data_List[0].madis_data[['lat']].to_dataarray().values.T)

    n_dataset = dict()

    loaders = dict()

    Train_Dataset = ConcatDataset(Data_List[:int(n_years * 0.7)])
    Valid_Dataset = ConcatDataset(Data_List[int(n_years * 0.7):int(n_years * 0.7) + int(n_years * 0.2)])
    Test_Dataset = ConcatDataset(Data_List[int(n_years * 0.7) + int(n_years * 0.2):])

    loaders['train'] = DataLoader(Train_Dataset, batch_size=batch_size, shuffle=True)
    loaders['val'] = DataLoader(Valid_Dataset, batch_size=batch_size, shuffle=True)
    loaders['test'] = DataLoader(Test_Dataset, batch_size=batch_size, shuffle=False)

    n_dataset['train'] = len(Train_Dataset)
    n_dataset['val'] = len(Valid_Dataset)
    n_dataset['test'] = len(Test_Dataset)

    n_stations = madis_network.n_stations
    n_train_stations = madis_network.n_stations
    n_val_stations = madis_network.n_stations
    n_test_stations = madis_network.n_stations

    madis_norm_dict, era5_norm_dict = NormalizerBuilder.get_normalizers(Data_List, era5_network)

    print('n_stations: ', n_stations, flush=True)

    ##### Define Model #####
    if model_type == ModelType.MLP:
        model = PlainMLP(Madis_len,
                         ERA5_len if era5_network is not None else None,
                         hidden_dim=hidden_dim).to(device)

    elif model_type == ModelType.MPNN_MLP:
        model = MPNN_MLP(Madis_len,
                         ERA5_len if era5_network is not None else None,
                         2,
                         hidden_dim=hidden_dim).to(device)

    elif model_type == ModelType.GNN:
        model = MPNN(
            n_passing,
            lead_hrs=lead_hrs,
            n_node_features_m=3 * Madis_len,
            n_node_features_e=3 * ERA5_len,
            n_out_features=2,
            hidden_dim=hidden_dim
        ).to(device)

    else:
        raise NotImplementedError

    nn_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('Parameter Number: ', nn_params, flush=True)
    print(' ', flush=True)

    loss_function = nn.MSELoss(reduction='sum')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    ###### Training ######
    train_losses = []
    valid_losses = []
    test_losses = []
    test_u_mses = []
    test_u_maes = []
    test_v_mses = []
    test_v_maes = []

    min_valid_loss = 9999999999

    for epoch in range(epochs):
        test_epoch = (epoch + 1) % eval_interval == 0 or epoch == 0

        call_evaluate = lambda dataset, save: evaluate_model(
            model,
            loaders[dataset],
            madis_norm_dict,
            era5_norm_dict,
            device,
            lead_hrs,
            loss_function=loss_function,
            optimizer=optimizer,
            save=save,
            model_type=model_type,
            station_type=dataset,
            show_progress_bar=show_progress_bar
        )

        if test_epoch:
            MAE_u_sum, MSE_u_sum, MAE_v_sum, MSE_v_sum, Pred_train, Target_train = call_evaluate('train', True)

        else:
            MAE_u_sum, MSE_u_sum, MAE_v_sum, MSE_v_sum = call_evaluate('train', False)

        train_loss = (MSE_u_sum + MSE_v_sum) / (n_dataset['train'] * n_train_stations)
        train_losses.append(train_loss)

        if test_epoch:
            MAE_u_sum, MSE_u_sum, MAE_v_sum, MSE_v_sum, Pred_val, Target_val = call_evaluate('val', True)
        else:
            MAE_u_sum, MSE_u_sum, MAE_v_sum, MSE_v_sum = call_evaluate('val', False)

        valid_loss = (MSE_u_sum + MSE_v_sum) / (n_dataset['val'] * n_val_stations)
        valid_losses.append(valid_loss)

        print('Epoch: %d train_loss[%.3f] valid_loss[%.3f]' % (epoch + 1, train_loss, valid_loss), flush=True)
        print(' ', flush=True)

        if test_epoch:
            MAE_u_sum, MSE_u_sum, MAE_v_sum, MSE_v_sum, Pred, Target = call_evaluate('test', True)

            Preds = dict()
            Preds['train'] = Pred_train
            Preds['val'] = Pred_val
            Preds['test'] = Pred

            Targets = dict()
            Targets['train'] = Target_train
            Targets['val'] = Target_val
            Targets['test'] = Target

            test_u_mae = MAE_u_sum / (n_dataset['test'] * n_test_stations)
            test_u_mse = MSE_u_sum / (n_dataset['test'] * n_test_stations)

            test_v_mae = MAE_v_sum / (n_dataset['test'] * n_test_stations)
            test_v_mse = MSE_v_sum / (n_dataset['test'] * n_test_stations)

            test_u_maes.append(test_u_mae)
            test_u_mses.append(test_u_mse)

            test_v_maes.append(test_v_mae)
            test_v_mses.append(test_v_mse)

            test_loss = (MSE_u_sum + MSE_v_sum) / (n_dataset['test'] * n_test_stations)
            test_loss_mae = (MAE_u_sum + MAE_v_sum) / (n_dataset['test'] * n_test_stations)
            test_losses.append(test_loss)

            # if (epoch + 1) == epochs:
            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss

                serialized_data = pickle.dumps(madis_network)
                # Step 4: Write the serialized data to a file
                with open(output_saving_path / f'madis_network_min.pkl', 'wb') as file:
                    file.write(serialized_data)

                serialized_data = pickle.dumps(Targets)
                # Step 4: Write the serialized data to a file
                with open(output_saving_path / f'Targets_min.pkl', 'wb') as file:
                    file.write(serialized_data)

                serialized_data = pickle.dumps(Preds)
                # Step 4: Write the serialized data to a file
                with open(output_saving_path / f'Preds_min.pkl', 'wb') as file:
                    file.write(serialized_data)

                np.save(os.path.join(output_saving_path, f'min_test_loss_mse.npy'), test_loss)
                np.save(os.path.join(output_saving_path, f'min_test_loss_mae.npy'), test_loss_mae)
                np.save(os.path.join(output_saving_path, f'min_test_u_mae.npy'), test_u_mae)
                np.save(os.path.join(output_saving_path, f'min_test_v_mae.npy'), test_v_mae)
                np.save(os.path.join(output_saving_path, f'min_test_u_mse.npy'), test_u_mse)
                np.save(os.path.join(output_saving_path, f'min_test_v_mse.npy'), test_v_mse)

            print('Evaluation Report: test_u_mae[%.3f] test_u_mse[%.3f] test_v_mae[%.3f] test_v_mse[%.3f]' % (
                test_u_mae, test_u_mse, test_v_mae, test_v_mse), flush=True)
            print(' ', flush=True)

            np.save(os.path.join(output_saving_path, f'station_train_test_u_mae_epoch_{epoch + 1}.npy'),
                    MAE_u_sum / n_dataset['test'])
            np.save(os.path.join(output_saving_path, f'station_train_test_u_mse_epoch_{epoch + 1}.npy'),
                    MSE_u_sum / n_dataset['test'])
            np.save(os.path.join(output_saving_path, f'station_train_test_v_mae_epoch_{epoch + 1}.npy'),
                    MAE_v_sum / n_dataset['test'])
            np.save(os.path.join(output_saving_path, f'station_train_test_v_mse_epoch_{epoch + 1}.npy'),
                    MSE_v_sum / n_dataset['test'])
            np.save(os.path.join(output_saving_path, f'station_train_test_preds_epoch_{epoch + 1}.npy'), Pred)

            torch.save(model.state_dict(), os.path.join(output_saving_path, f'model_epoch_{epoch + 1}.pt'))

    ##### Save #####
    train_losses = np.array(train_losses)
    np.save(os.path.join(output_saving_path, 'train_losses.npy'), train_losses)

    valid_losses = np.array(valid_losses)
    np.save(os.path.join(output_saving_path, 'valid_losses.npy'), valid_losses)

    test_u_mses = np.array(test_u_mses)
    np.save(os.path.join(output_saving_path, 'test_u_mses.npy'), test_u_mses)

    test_u_maes = np.array(test_u_maes)
    np.save(os.path.join(output_saving_path, 'test_u_maes.npy'), test_u_maes)

    test_v_mses = np.array(test_v_mses)
    np.save(os.path.join(output_saving_path, 'test_v_mses.npy'), test_v_mses)

    test_v_maes = np.array(test_v_maes)
    np.save(os.path.join(output_saving_path, 'test_v_maes.npy'), test_v_maes)

    ##### Plotting #####
    plot_metric(train_losses, valid_losses, test_losses, eval_interval, 'MSE', figures_path)


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
