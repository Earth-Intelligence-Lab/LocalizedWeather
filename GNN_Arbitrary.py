# This file is used to train wind correction model and evaluate it.
# Author: Qidong Yang
# Date: 2024-02-14

import argparse
import pickle
from pathlib import Path
import geopandas as gpd
import numpy as np

import torch

from Dataloader.ERA5 import ERA5
from Dataloader.MetaStation import MetaStation
from Dataloader.MixDataArbitrary import MixDataArbitrary
from Dataloader.MixDataArbitraryMLP import MixDataArbitraryMLP
from Modules.GNN.MPNNArbitraryLocation import MPNNArbitraryLocation
from Modules.MLP.PlainMLPArbitrary import PlainMLPArbitrary
from Network.ERA5Network import ERA5Network
from Network.MadisNetwork import MadisNetwork
from Settings.Settings import TemporalDatasetSplit, StationSamplingMethod, StaticNodeType, ModelType
from torch.utils.data import DataLoader, ConcatDataset

import Stats.MadisInterpolation as MadisInterpolation
import Stats.ERA5Interpolation as ERA5Interpolation

from evaluationArbitraryLocation import evaluationArbitraryLocation
from utilities import MaxMinNormalizer, ABNormalizer

def Run(args):
    root_path = Path(args.root_path)
    output_saving_path = root_path/args.output_saving_path
    output_saving_path.mkdir(exist_ok=True, parents=True)
    show_progress_bar = args.show_progress_bar
    shapefile_path = args.shapefile_path

    if shapefile_path is None:
        lon_low, lon_up, lat_low, lat_up = args.coords
    else:
        shapefile_path = root_path/shapefile_path
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

    figures_path = output_saving_path/'figures'
    figures_path.mkdir(exist_ok=True, parents=True)

    print('Experiment Configuration', flush=True)

    for k, v in vars(args).items():
        print(f'{k}: {v}', flush=True)


    ##### Set Random Seed #####
    np.random.seed(100)


    ##### Get Device #####
    device = 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    ##### Load Data #####
    meta_station = MetaStation(lat_low, lat_up, lon_low, lon_up, n_years, madis_control_ratio, shapefile_path=shapefile_path, root_path=root_path)

    madis_network = MadisNetwork(meta_station, station_sampling_method, p_arbitrary_nodes, p_arbitrary_nodes_valid, p_arbitrary_nodes_test, n_neighbors_m2m, network_construction_method)

    if n_neighbors_e2m > 0:
        era5_stations = ERA5(meta_station.lat_low, meta_station.lat_up, meta_station.lon_low, meta_station.lon_up, 2023,
             root_path=root_path).data

        era5_network = ERA5Network(era5_stations, madis_network, n_neighbors_e2m)
    else:
        era5_network = None

    years = list(range(2024-n_years, 2024))



    # for dataset in ['train', 'val', 'test']:
    if model_type == ModelType.GNN:
        Data_List = [MixDataArbitrary(year, back_hrs, lead_hrs, meta_station, madis_network, static_node, n_neighbors_m2m, era5_network, 'train',
                                          edge_type=edge_type, root_path=root_path) for year in years]
    else:
        Data_List = [MixDataArbitraryMLP(year, back_hrs, lead_hrs, meta_station, madis_network, static_node, n_neighbors_m2m, era5_network, 'train',
                                         edge_type=edge_type, root_path=root_path) for year in years]

        if era5_network is not None:
            era5_network.era5_pos = torch.Tensor(Data_List[0].madis_data[['lon', 'lat']].to_dataarray().values.T)



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



    ##### Get Normalizer #####
    madis_u_mins = []
    madis_v_mins = []
    madis_temp_mins = []
    madis_elv_mins = []

    madis_u_maxs = []
    madis_v_maxs = []
    madis_temp_maxs = []
    madis_elv_maxs = []

    for Data in Data_List:
        madis_u_mins.append(Data.madis_u_min)
        madis_v_mins.append(Data.madis_v_min)
        madis_temp_mins.append(Data.madis_temp_min)
        madis_elv_mins.append(Data.madis_elv_min)

        madis_u_maxs.append(Data.madis_u_max)
        madis_v_maxs.append(Data.madis_v_max)
        madis_temp_maxs.append(Data.madis_temp_max)
        madis_elv_maxs.append(Data.madis_elv_max)

    madis_temp_min = np.min(np.array(madis_temp_mins))
    madis_elv_min = np.min(np.array(madis_elv_mins))
    madis_temp_max = np.max(np.array(madis_temp_maxs))
    madis_elv_max = np.max(np.array(madis_elv_maxs))

    madis_temp_normalizer = MaxMinNormalizer(madis_temp_min, madis_temp_max)

    madis_u_min = np.min(np.array(madis_u_mins))
    madis_v_min = np.min(np.array(madis_v_mins))
    madis_u_max = np.max(np.array(madis_u_maxs))
    madis_v_max = np.max(np.array(madis_v_maxs))
    madis_scale = np.max(np.array([np.abs(madis_u_min), np.abs(madis_v_min), madis_u_max, madis_v_max]))

    madis_u_normalizer = ABNormalizer(-madis_scale, madis_scale, -1.0, 1.0)
    madis_v_normalizer = ABNormalizer(-madis_scale, madis_scale, -1.0, 1.0)
    madis_norm_dict = {'u': madis_u_normalizer, 'v': madis_v_normalizer, 'temp': madis_temp_normalizer}


    if era5_network is not None:
        era5_u_mins = []
        era5_v_mins = []
        era5_temp_mins = []

        era5_u_maxs = []
        era5_v_maxs = []
        era5_temp_maxs = []

        for Data in Data_List:

            era5_u_mins.append(Data.era5_u_min)
            era5_v_mins.append(Data.era5_v_min)
            era5_temp_mins.append(Data.era5_temp_min)

            era5_u_maxs.append(Data.era5_u_max)
            era5_v_maxs.append(Data.era5_v_max)
            era5_temp_maxs.append(Data.era5_temp_max)

        era5_u_min = np.min(np.array(era5_u_mins))
        era5_v_min = np.min(np.array(era5_v_mins))
        era5_temp_min = np.min(np.array(era5_temp_mins))
        era5_u_max = np.max(np.array(era5_u_maxs))
        era5_v_max = np.max(np.array(era5_v_maxs))
        era5_temp_max = np.max(np.array(era5_temp_maxs))
        era5_scale = np.max(np.array([np.abs(era5_u_min), np.abs(era5_v_min), era5_u_max, era5_v_max]))

        era5_u_normalizer = MaxMinNormalizer(-era5_scale, era5_scale)
        era5_v_normalizer = MaxMinNormalizer(-era5_scale, era5_scale)

        era5_temp_normalizer = MaxMinNormalizer(era5_temp_min, era5_temp_max)

        era5_norm_dict = {'u': era5_u_normalizer, 'v': era5_v_normalizer, 'temp': era5_temp_normalizer}
    else:
        era5_norm_dict = None

    #### Split Dataset #####
    # n_edge_features = Data_List[0].n_edge_features

    n_stations = Data_List[0].n_stations
    stat_lats = Data_List[0].stat_lats.reshape((-1, 1))
    stat_lons = Data_List[0].stat_lons.reshape((-1, 1))

    if madis_network.station_sampling_method == StationSamplingMethod.none:
        train_stat_ind = madis_network.ind_orig_stations_logical_known['train']
        val_stat_ind = madis_network.ind_orig_stations_logical_known['val']
        test_stat_ind = madis_network.ind_orig_stations_logical_known['test']
    else:
        train_stat_ind = madis_network.ind_orig_stations_logical_unknown['train']
        val_stat_ind = madis_network.ind_orig_stations_logical_unknown['val']
        test_stat_ind = madis_network.ind_orig_stations_logical_unknown['test']

    if madis_network.station_sampling_method == StationSamplingMethod.none:
        n_train_stations = madis_network.n_stations
    elif madis_network.station_sampling_method == StationSamplingMethod.random_hold_out:
        n_train_stations = madis_network.m_arbitrary_nodes
    else:
        n_train_stations = np.sum(madis_network.ind_orig_stations_logical_unknown['train'])

    if madis_network.station_sampling_method != StationSamplingMethod.none:
        n_val_stations = np.sum(madis_network.ind_orig_stations_logical_unknown['val'])
        n_test_stations = np.sum(madis_network.ind_orig_stations_logical_unknown['test'])
    else:
        n_val_stations = madis_network.n_stations
        n_test_stations = madis_network.n_stations


    np.save(os.path.join(output_saving_path, 'train_stat_ind.npy'), train_stat_ind)
    np.save(os.path.join(output_saving_path, 'val_stat_ind.npy'), val_stat_ind)
    np.save(os.path.join(output_saving_path, 'test_stat_ind.npy'), test_stat_ind)
    np.save(os.path.join(output_saving_path, 'stat_locations.npy'), np.concatenate([stat_lons, stat_lats], axis=1))


    print('n_stations: ', n_stations, flush=True)
    print('n_station_train: ', n_train_stations, flush=True)
    print('n_station_val: ', n_val_stations, flush=True)
    print('n_station_test: ', n_test_stations, flush=True)
    # print('train_set_size: ', n_train, flush=True)
    # print('valid_set_size: ', n_valid, flush=True)
    # print('test_set_size: ', n_test, flush=True)
    # print('n_edge_features: ', n_edge_features, flush=True)

    ##### Define Model #####
    if model_type == ModelType.MLP:
        model = PlainMLPArbitrary(Madis_len,
                                  ERA5_len if era5_network is not None else None,
                                  static_node=static_node,
                                  static_node_size=static_node_image_size,
                                  hidden_dim=hidden_dim).to(device)
    elif model_type == ModelType.GNN:
        model = MPNNArbitraryLocation(
            n_passing,
            n_stations,
            lead_hrs = lead_hrs,
            n_variables_m=3,
            n_node_features_m=3*Madis_len,
            n_node_features_e=3*ERA5_len,
            n_edge_features=0,
            interpolate=interpolate,
            static_node_size=static_node_image_size,
            static_node=static_node,
            n_out_features=2,
            hidden_dim=hidden_dim
        ).to(device)
    else:
        raise NotImplementedError

    nn_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('Parameter Number: ', nn_params, flush=True)
    print(' ', flush=True)

    extrapolationLossFunction = nn.MSELoss(reduction='mean')
    interpolationLossFunction = nn.MSELoss(reduction='mean')
    # myloss = nn.MSELoss(reduction='sum')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


    ###### Training ######
    train_losses = []
    valid_losses = []
    test_losses = []
    test_u_mses = []
    test_u_maes = []
    test_v_mses = []
    test_v_maes = []

    # if station_held != 'none':
    #     test_u_mses_held = []
    #     test_u_maes_held = []
    #     test_v_mses_held = []
    #     test_v_maes_held = []

    min_valid_loss = 9999999999

    for epoch in range(epochs):
        test_epoch = (epoch + 1) % eval_interval == 0 or epoch == 0

        if test_epoch:
            MAE_u_sum, MSE_u_sum, MAE_v_sum, MSE_v_sum, Pred_train, Target_train = evaluationArbitraryLocation(epoch,
                                                                                                               model,
                                                                                                               loaders['train'],
                                                                                                               madis_network,
                                                                                                               madis_norm_dict,
                                                                                                               era5_norm_dict,
                                                                                                               static_node_type,
                                                                                                               device,
                                                                                                               lead_hrs,
                                                                                                               interpolate,
                                                                                                               interpolationLossFunction=interpolationLossFunction,
                                                                                                               loss_function=extrapolationLossFunction,
                                                                                                               loss_factor_interpolation=loss_factor_interpolate,
                                                                                                               loss_factor_extrapolate_known=loss_factor_extrapolate_known,
                                                                                                               loss_factor_extrapolate_unknown=loss_factor_extrapolate_unknown,
                                                                                                               optimizer=optimizer,
                                                                                                               save=True,
                                                                                                               model_type=model_type,
                                                                                                               station_type='train',
                                                                                                               show_progress_bar=show_progress_bar,
                                                                                                               figures_output_folder=figures_path)

        else:

            MAE_u_sum, MSE_u_sum, MAE_v_sum, MSE_v_sum = evaluationArbitraryLocation(epoch,
                                                                                     model,
                                                                                     loaders['train'],
                                                                                     madis_network,
                                                                                     madis_norm_dict,
                                                                                     era5_norm_dict,
                                                                                     static_node_type,
                                                                                     device,
                                                                                     lead_hrs,
                                                                                     interpolate,
                                                                                     interpolationLossFunction=interpolationLossFunction,
                                                                                     loss_function=extrapolationLossFunction,
                                                                                     loss_factor_interpolation=loss_factor_interpolate,
                                                                                     loss_factor_extrapolate_known=loss_factor_extrapolate_known,
                                                                                     loss_factor_extrapolate_unknown=loss_factor_extrapolate_unknown,
                                                                                     optimizer=optimizer,
                                                                                     save=False,
                                                                                     model_type=model_type,
                                                                                     station_type='train',
                                                                                     show_progress_bar=show_progress_bar,
                                                                                     figures_output_folder=figures_path)


        train_loss = (MSE_u_sum + MSE_v_sum) / (n_dataset['train'] * n_train_stations)


        if test_epoch:
            MAE_u_sum, MSE_u_sum, MAE_v_sum, MSE_v_sum, Pred_val, Target_val = evaluationArbitraryLocation(epoch,
                                                                                                           model,
                                                                                                           loaders['val'],
                                                                                                           madis_network,
                                                                                                           madis_norm_dict,
                                                                                                           era5_norm_dict,
                                                                                                           static_node_type,
                                                                                                           device,
                                                                                                           lead_hrs,
                                                                                                           interpolate,
                                                                                                           save=True,
                                                                                                           model_type=model_type,
                                                                                                           station_type='val',
                                                                                                           show_progress_bar=show_progress_bar)
        else:
            MAE_u_sum, MSE_u_sum, MAE_v_sum, MSE_v_sum = evaluationArbitraryLocation(epoch,
                                                                                     model,
                                                                                     loaders['val'],
                                                                                     madis_network,
                                                                                     madis_norm_dict,
                                                                                     era5_norm_dict,
                                                                                     static_node_type,
                                                                                     device,
                                                                                     lead_hrs,
                                                                                     interpolate,
                                                                                     save=False,
                                                                                     model_type=model_type,
                                                                                     station_type='val',
                                                                                     show_progress_bar=show_progress_bar)
        # (n_stations, 1)
        valid_loss = (MSE_u_sum + MSE_v_sum) / (n_dataset['val'] * n_val_stations)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print('Epoch: %d train_loss[%.3f] valid_loss[%.3f]' % (epoch + 1, train_loss, valid_loss), flush=True)
        print(' ', flush=True)

        if test_epoch:
            MAE_u_sum, MSE_u_sum, MAE_v_sum, MSE_v_sum, Pred, Target = evaluationArbitraryLocation(epoch,
                                                                                                   model,
                                                                                                   loaders['test'],
                                                                                                   madis_network,
                                                                                                   madis_norm_dict,
                                                                                                   era5_norm_dict,
                                                                                                   static_node_type,
                                                                                                   device,
                                                                                                   lead_hrs,
                                                                                                   interpolate,
                                                                                                   save=True,
                                                                                                   model_type=model_type,
                                                                                                   station_type='test',
                                                                                                   show_progress_bar=show_progress_bar)

            Preds = dict()
            Preds['train'] = Pred_train
            Preds['val'] = Pred_val
            Preds['test'] = Pred

            Targets = dict()
            Targets['train'] = Target_train
            Targets['val'] = Target_val
            Targets['test'] = Target

            # ArbitraryGNNErrors.PlotArbitraryGNNErrorsGraph(madis_network, Pred_val, Target_val, Pred, Target, epoch, figures_path)
            # ArbitraryGNNErrors.PlotScatterPlot(madis_network, Preds, Targets, epoch, figures_path)

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
                with open(output_saving_path/f'madis_network_min.pkl', 'wb') as file:
                    file.write(serialized_data)

                serialized_data = pickle.dumps(Targets)
                # Step 4: Write the serialized data to a file
                with open(output_saving_path/f'Targets_min.pkl', 'wb') as file:
                    file.write(serialized_data)

                serialized_data = pickle.dumps(Preds)
                # Step 4: Write the serialized data to a file
                with open(output_saving_path/f'Preds_min.pkl', 'wb') as file:
                    file.write(serialized_data)

                np.save(os.path.join(output_saving_path, f'min_test_loss_mse.npy'), test_loss)
                np.save(os.path.join(output_saving_path, f'min_test_loss_mae.npy'), test_loss_mae)
                np.save(os.path.join(output_saving_path, f'min_test_u_mae.npy'), test_u_mae)
                np.save(os.path.join(output_saving_path, f'min_test_v_mae.npy'), test_v_mae)
                np.save(os.path.join(output_saving_path, f'min_test_u_mse.npy'), test_u_mse)
                np.save(os.path.join(output_saving_path, f'min_test_v_mse.npy'), test_v_mse)

            print('Evaluation Report: test_u_mae[%.3f] test_u_mse[%.3f] test_v_mae[%.3f] test_v_mse[%.3f]' % (test_u_mae, test_u_mse, test_v_mae, test_v_mse), flush=True)
            print(' ', flush=True)

            np.save(os.path.join(output_saving_path, f'station_train_test_u_mae_epoch_{epoch + 1}.npy'), MAE_u_sum / n_dataset['test'])
            np.save(os.path.join(output_saving_path, f'station_train_test_u_mse_epoch_{epoch + 1}.npy'), MSE_u_sum / n_dataset['test'])
            np.save(os.path.join(output_saving_path, f'station_train_test_v_mae_epoch_{epoch + 1}.npy'), MAE_v_sum / n_dataset['test'])
            np.save(os.path.join(output_saving_path, f'station_train_test_v_mse_epoch_{epoch + 1}.npy'), MSE_v_sum / n_dataset['test'])
            np.save(os.path.join(output_saving_path, f'station_train_test_preds_epoch_{epoch + 1}.npy'), Pred)

            # if station_held != 'none':
            #     MAE_u_sum, MSE_u_sum, MAE_v_sum, MSE_v_sum, Pred = evaluationArbitraryLocation(model, test_loader, madis_norm_dict, device, lead_hrs, True, model_type=model_type, station_type='held')
            #
            #     test_u_mae_held = np.sum(MAE_u_sum) / (n_test * np.sum(held_stat_ind))
            #     test_u_mse_held = np.sum(MSE_u_sum) / (n_test * np.sum(held_stat_ind))
            #
            #     test_v_mae_held = np.sum(MAE_v_sum) / (n_test * np.sum(held_stat_ind))
            #     test_v_mse_held = np.sum(MSE_v_sum) / (n_test * np.sum(held_stat_ind))
            #
            #     test_u_maes_held.append(test_u_mae_held)
            #     test_u_mses_held.append(test_u_mse_held)
            #
            #     test_v_maes_held.append(test_v_mae_held)
            #     test_v_mses_held.append(test_v_mse_held)
            #
            #     print('Evaluation Report: test_u_mae_held[%.3f] test_u_mse_held[%.3f] test_v_mae_held[%.3f] test_v_mse_held[%.3f]' % (test_u_mae_held, test_u_mse_held, test_v_mae_held, test_v_mse_held), flush=True)
            #     print(' ', flush=True)
            #
            #     np.save(os.path.join(output_saving_path, f'station_held_test_u_mae_epoch_{epoch + 1}.npy'), MAE_u_sum / n_test)
            #     np.save(os.path.join(output_saving_path, f'station_held_test_u_mse_epoch_{epoch + 1}.npy'), MSE_u_sum / n_test)
            #     np.save(os.path.join(output_saving_path, f'station_held_test_v_mae_epoch_{epoch + 1}.npy'), MAE_v_sum / n_test)
            #     np.save(os.path.join(output_saving_path, f'station_held_test_v_mse_epoch_{epoch + 1}.npy'), MSE_v_sum / n_test)
            #     np.save(os.path.join(output_saving_path, f'station_held_test_preds_epoch_{epoch + 1}.npy'), Pred)

            torch.save(model.state_dict(), os.path.join(output_saving_path, f'model_epoch_{epoch + 1}.pt'))

    '''
        if epoch == 0:
            era5_MAE_sum, era5_MSE_sum, ERA5_DATA, MADIS_DATA = eval_ERA5(test_loader, np.sum(train_ind), 'train')
            np.save(os.path.join(output_saving_path, f'era5_station_mae.npy'), era5_MAE_sum / n_test)
            np.save(os.path.join(output_saving_path, f'era5_station_mse.npy'), era5_MSE_sum / n_test)
            np.save(os.path.join(output_saving_path, f'test_era5_data.npy'), ERA5_DATA)
            np.save(os.path.join(output_saving_path, f'test_madis_data.npy'), MADIS_DATA)
            
            if station_held != 'none':
                era5_MAE_sum, era5_MSE_sum, ERA5_DATA, MADIS_DATA = eval_ERA5(test_loader, np.sum(held_ind), 'held')
                np.save(os.path.join(output_saving_path, f'era5_station_mae_held.npy'), era5_MAE_sum / n_test)
                np.save(os.path.join(output_saving_path, f'era5_station_mse_held.npy'), era5_MSE_sum / n_test)
                np.save(os.path.join(output_saving_path, f'test_era5_data_held.npy'), ERA5_DATA)
                np.save(os.path.join(output_saving_path, f'test_madis_data_held.npy'), MADIS_DATA)
    '''

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

    # if station_held != 'none':
    #     test_u_mses_held = np.array(test_u_mses_held)
    #     np.save(os.path.join(output_saving_path, 'test_u_mses_held.npy'), test_u_mses_held)
    #
    #     test_u_maes_held = np.array(test_u_maes_held)
    #     np.save(os.path.join(output_saving_path, 'test_u_maes_held.npy'), test_u_maes_held)
    #
    #     test_v_mses_held = np.array(test_v_mses_held)
    #     np.save(os.path.join(output_saving_path, 'test_v_mses_held.npy'), test_v_mses_held)
    #
    #     test_v_maes_held = np.array(test_v_maes_held)
    #     np.save(os.path.join(output_saving_path, 'test_v_maes_held.npy'), test_v_maes_held)

    ##### Plotting #####
    plot_metric(train_losses, valid_losses, test_losses, eval_interval, 'MSE', figures_path)

