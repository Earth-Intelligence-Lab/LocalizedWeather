# Author: Qidong Yang & Jonathan Giezendanner

import os
import pickle
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset

import Normalization.NormalizerBuilder as NormalizerBuilder
from Dataloader.ERA5 import ERA5
from Dataloader.ERA5Interpolated import ERA5Interpolated
from Dataloader.HRRR import HRRR
from Dataloader.HRRRInterpolated import HRRRInterpolated
from Dataloader.MetaStation import MetaStation
from Dataloader.MixData import MixData
from EvaluateModel import EvaluateModel
from Modules.GNN.MPNN import MPNN
from Utils.LossFunctions import GetLossFunctionReport, GetLossFunction, SetupSaveMetrics
from Modules.Transformer.ViT import VisionTransformer
from Network.ERA5Network import ERA5Network
from Network.HRRRNetwork import HRRNetwork
from Network.MadisNetwork import MadisNetwork
from Settings.Settings import ModelType, LossFunctionType, InterpolationType
from Utils.Telemetry import Telemetry, WBTelemetry


class Main:
    def __init__(self, args):
        self.args = args
        self.data_path = Path(args.data_path)
        self.output_saving_path = self.data_path / args.output_saving_path
        self.output_saving_path.mkdir(exist_ok=True, parents=True)
        self.show_progress_bar = args.show_progress_bar
        self.shapefile_path = args.shapefile_path

        if self.shapefile_path is None:
            self.lon_low, self.lon_up, self.lat_low, self.lat_up = args.coords
        else:
            self.shapefile_path = self.data_path / self.shapefile_path
            self.lon_low, self.lat_low, self.lon_up, self.lat_up = gpd.read_file(self.shapefile_path).bounds.iloc[0].values

        self.back_hrs = args.back_hrs
        self.lead_hrs = args.lead_hrs
        self.whole_len = self.back_hrs + 1

        self.Madis_len = self.whole_len
        self.external_len = self.whole_len + self.lead_hrs
        self.hidden_dim = args.hidden_dim
        self.lr = args.lr
        self.loss_function_type = args.loss_function_type
        self.save_metrics_types = [LossFunctionType.CUSTOM]
        self.per_variable_metrics_types = [LossFunctionType.MSE, LossFunctionType.MAE]
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.weight_decay = args.weight_decay
        self.model_type = args.model_type
        self.madis_control_ratio = args.madis_control_ratio
        self.n_years = args.n_years
        self.n_passing = args.n_passing
        self.n_neighbors_m2m = args.n_neighbors_m2m
        self.network_construction_method = args.network_construction_method

        self.n_neighbors_e2m = args.n_neighbors_e2m
        self.n_neighbors_h2m = args.n_neighbors_h2m
        self.hrrr_analysis_only = args.hrrr_analysis_only

        self.interpolation_type = args.interpolation_type

        self.madis_vars_i = args.madis_vars_i
        self.madis_vars_o = args.madis_vars_o
        self.madis_vars = list(set(self.madis_vars_i + self.madis_vars_o))

        self.external_vars = args.external_vars

        self.figures_path = self.output_saving_path / 'figures'
        self.figures_path.mkdir(exist_ok=True, parents=True)

        self.use_wb = args.use_wb

        print('Experiment Configuration', flush=True)

        for k, v in vars(args).items():
            print(f'{k}: {v}', flush=True)

    def Run(self):

        ##### Set Random Seed #####
        np.random.seed(100)

        ##### Get Device #####
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

        ##### Load Data #####
        meta_station = MetaStation(self.lat_low, self.lat_up, self.lon_low, self.lon_up, self.n_years, self.madis_control_ratio,
                                   shapefile_path=self.shapefile_path, data_path=self.data_path)

        self.madis_network = self.BuildMadisNetwork(meta_station, self.n_neighbors_m2m, self.network_construction_method)

        years = list(range(2024 - self.n_years, 2024))

        external_data_objects, self.external_len, self.external_network, n_neighbors_ex2m = self.BuildExternalNetwork(
                                                                                     self.data_path,
                                                                                     self.external_len,
                                                                                     self.hrrr_analysis_only,
                                                                                     self.interpolation_type, self.lead_hrs,
                                                                                     self.madis_network, meta_station,
                                                                                     self.n_neighbors_e2m, self.n_neighbors_h2m,
                                                                                     self.whole_len, years)

        Data_List = self.GetDataList(self.back_hrs, self.data_path, external_data_objects, self.external_network, self.external_vars,
                                     self.lead_hrs, self.madis_vars, meta_station, years)

        if external_data_objects:
            for k, year in enumerate(years):
                external_data_objects[year].LoadDataToMemory()

        loaders = self.CreateDataLoaders(Data_List, self.batch_size, self.n_years)

        n_stations = self.madis_network.n_stations

        madis_norm_dict, external_norm_dict = NormalizerBuilder.get_normalizers(Data_List, external_data_objects, self.madis_vars, self.external_vars)

        serialized_data = pickle.dumps(madis_norm_dict)
        with open(self.output_saving_path / f'madis_norm_dict.pkl', 'wb') as file:
            file.write(serialized_data)

        serialized_data = pickle.dumps(external_norm_dict)
        with open(self.output_saving_path / f'external_norm_dict.pkl', 'wb') as file:
            file.write(serialized_data)

        print('n_stations: ', n_stations, flush=True)

        model = self.GetModel(self.Madis_len, self.external_len, external_data_objects is not None, self.external_vars,
                              self.hidden_dim, self.lead_hrs, self.madis_vars_i, self.madis_vars_o, self.model_type,
                              self.n_passing, n_stations, self.madis_network).to(device)

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        nn_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print('Parameter Number: ', nn_params, flush=True)
        print(' ', flush=True)

        loss_function = GetLossFunction(self.loss_function_type, self.madis_vars_o)
        loss_function_report = GetLossFunctionReport(self.loss_function_type, self.madis_vars_o)
        save_metrics = SetupSaveMetrics(self.save_metrics_types, self.madis_vars_o)

        use_wb = self.use_wb
        if use_wb:
            telemetry = WBTelemetry(self.madis_vars_o, self.per_variable_metrics_types, self.args, self.output_saving_path)
        else:
            telemetry = Telemetry(self.madis_vars_o, self.per_variable_metrics_types)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)


        min_valid_losses = dict()
        for save_metric_type in self.save_metrics_types:
            min_valid_losses[save_metric_type] = 9999999999

        evaluateModel_fun = lambda: self.GetEvaluateModel(device, external_norm_dict, self.external_vars, self.lead_hrs,
                                                          loaders, loss_function, loss_function_report, madis_norm_dict,
                                                          self.madis_vars, self.madis_vars_i, self.madis_vars_o, model,
                                                          self.model_type, optimizer, telemetry.per_variable_metrics,
                                                          self.per_variable_metrics_types, save_metrics,
                                                          self.save_metrics_types, self.show_progress_bar)

        for epoch in range(self.epochs):
            evaluateModel = evaluateModel_fun()

            train_loss, per_variable_loss_train = evaluateModel.call_evaluate('train', False)

            telemetry.addLoss(train_loss, per_variable_loss_train, 'train')

            evaluateModel = evaluateModel_fun()
            valid_loss, per_variable_loss_valid = evaluateModel.call_evaluate('val', False)

            telemetry.addLoss(valid_loss, per_variable_loss_valid, 'val')

            telemetry.report(epoch, self.lr)

            for save_metric_type in self.save_metrics_types:
                min_valid_losses[save_metric_type] = self.SaveModel(model, min_valid_losses[save_metric_type],
                                                                    evaluateModel.save_metric_dict[save_metric_type],
                                                                    save_metric_type.name, self.output_saving_path)

        best_metrics = self.AtTrainingEnd(self.save_metrics_types, evaluateModel_fun, self.madis_network, self.output_saving_path)

        telemetry.finish_run(best_metrics, self.figures_path)

        print('Done')

    def GetModel(self, Madis_len, external_len, use_external, external_vars, hidden_dim, lead_hrs, madis_vars_i,
                 madis_vars_o, model_type, n_passing, n_stations, madis_network):
        ##### Define Model #####

        if model_type == ModelType.GNN:

            if n_passing == -4:
                import networkx as nx
                G = nx.Graph()
                G.add_edges_from(madis_network.k_edge_index.numpy().T)
                n_passing = nx.diameter(G)
                print(f"calculted n_passing from graph diameter: {n_passing}")

            model = MPNN(
                n_passing,
                lead_hrs=lead_hrs,
                n_node_features_m=len(madis_vars_i) * Madis_len,
                n_node_features_e=len(external_vars) * external_len,
                n_out_features=len(madis_vars_o),
                hidden_dim=hidden_dim
            )

        elif model_type == ModelType.ViT:
            model = VisionTransformer(n_stations,
                                      Madis_len,
                                      len(madis_vars_i),
                                      len(madis_vars_o),
                                      hidden_dim,  # dim
                                      hidden_dim // 2,  # attn_dim
                                      hidden_dim,  # mlp_dim
                                      3,  # num_heads
                                      5,  # num_layers
                                      len(external_vars),
                                      external_len if use_external else None
                                      )

        else:
            raise NotImplementedError
        return model

    def GetDataList(self, back_hrs, data_path, external_data_objects, external_network, external_vars, lead_hrs,
                    madis_vars, meta_station, years):
        Data_List = [MixData(year, back_hrs, lead_hrs, meta_station, self.madis_network, madis_vars, external_network,
                             external_vars, external_data_objects[year] if external_data_objects is not None else None,
                             data_path=data_path) for year in years]
        return Data_List

    def GetEvaluateModel(self, device, external_norm_dict, external_vars, lead_hrs, loaders, loss_function,
                         loss_function_report, madis_norm_dict, madis_vars, madis_vars_i, madis_vars_o, model,
                         model_type, optimizer, per_variable_metrics, per_variable_metrics_types, save_metrics,
                         save_metrics_types, show_progress_bar):
        evaluateModel = EvaluateModel(model, loaders, madis_norm_dict, external_norm_dict, device, lead_hrs,
                                      madis_vars_i, madis_vars_o, madis_vars, external_vars,
                                      loss_function=loss_function, loss_function_report=loss_function_report,
                                      save_metrics_types=save_metrics_types, save_metrics_functions=save_metrics,
                                      per_variable_metrics_types=per_variable_metrics_types,
                                      per_variable_metrics=per_variable_metrics, model_type=model_type,
                                      show_progress_bar=show_progress_bar, optimizer=optimizer)
        return evaluateModel

    def CreateDataLoaders(self, Data_List, batch_size, n_years):
        Train_Dataset = ConcatDataset(Data_List[:int(n_years * 0.7)])
        Valid_Dataset = ConcatDataset(Data_List[int(n_years * 0.7):int(n_years * 0.7) + int(n_years * 0.2)])
        Test_Dataset = ConcatDataset(Data_List[int(n_years * 0.7) + int(n_years * 0.2):])
        loaders = dict()
        loaders['train'] = DataLoader(Train_Dataset, batch_size=batch_size, shuffle=True)
        loaders['val'] = DataLoader(Valid_Dataset, batch_size=batch_size, shuffle=True)
        loaders['test'] = DataLoader(Test_Dataset, batch_size=batch_size, shuffle=False)
        n_dataset = dict()
        n_dataset['train'] = len(Train_Dataset)
        n_dataset['val'] = len(Valid_Dataset)
        n_dataset['test'] = len(Test_Dataset)
        return loaders

    def BuildExternalNetwork(self, data_path, external_len,
                             hrrr_analysis_only, interpolation_type, lead_hrs, madis_network, meta_station,
                             n_neighbors_e2m, n_neighbors_h2m, whole_len, years):


        external_data_objects = None
        external_network = None
        n_neighbors_ex2m = None

        if n_neighbors_e2m > 0:
            external_data_objects = dict()
            era5_loader = self.GetERA5Loader(interpolation_type)
            for year in years:
                if interpolation_type == InterpolationType.none:
                    external_data_objects[year] = era5_loader(year,
                                                              madis_network, meta_station, n_neighbors_e2m,
                                                              data_path=data_path)
                else:
                    external_data_objects[year] = era5_loader(year,
                                                              madis_network, meta_station, n_neighbors_e2m,
                                                              interpolation_type, data_path=data_path)

            if interpolation_type == InterpolationType.none:
                external_network = self.BuildERA5Network(external_data_objects[years[0]].data, madis_network, n_neighbors_e2m)

            n_neighbors_ex2m = n_neighbors_e2m

            if interpolation_type == InterpolationType.Stacked:
                external_len *= n_neighbors_e2m


        elif n_neighbors_h2m > 0:
            external_data_objects = dict()
            hrrr_loader = HRRRInterpolated if interpolation_type != InterpolationType.none else HRRR
            resampling_index_raw, resampling_index = None, None
            for year in years:
                external_data_objects[year] = hrrr_loader(meta_station, madis_network, year, hrrr_analysis_only,
                                                          resampling_index_raw, resampling_index,
                                                          data_path=data_path)
                resampling_index_raw = external_data_objects[year].resampling_index_raw
                resampling_index = external_data_objects[year].resampling_index

            external_network = HRRNetwork(external_data_objects[years[0]].data, madis_network, n_neighbors_h2m)

            if not hrrr_analysis_only:
                external_len = whole_len + np.min([lead_hrs, 18])

            n_neighbors_ex2m = n_neighbors_h2m

        return external_data_objects, external_len, external_network, n_neighbors_ex2m

    def GetERA5Loader(self, interpolation_type):
        return ERA5Interpolated if interpolation_type != InterpolationType.none else ERA5

    def BuildERA5Network(self, data, madis_network, n_neighbors_e2m):
        return ERA5Network(data, madis_network, n_neighbors_e2m)

    def BuildMadisNetwork(self, meta_station, n_neighbors_m2m, network_construction_method):
        return MadisNetwork(meta_station, n_neighbors_m2m, network_construction_method)

    def SaveModel(self, model, min_valid_loss, new_valid_loss, metric, output_saving_path):
        if (new_valid_loss >= min_valid_loss) or (np.isnan(new_valid_loss)):
            return min_valid_loss

        torch.save(model.state_dict(), os.path.join(output_saving_path, f'model_{metric}_min.pt'))

        return new_valid_loss


    def SaveTest(self, evaluateModelTest, madis_network, metric, output_saving_path):

        serialized_data = pickle.dumps(madis_network)
        with open(output_saving_path / f'madis_network.pkl', 'wb') as file:
            file.write(serialized_data)

        serialized_data = pickle.dumps(evaluateModelTest.Targets)
        with open(output_saving_path / f'Targets_{metric}_min.pkl', 'wb') as file:
            file.write(serialized_data)

        serialized_data = pickle.dumps(evaluateModelTest.Preds)
        with open(output_saving_path / f'Preds_{metric}_min.pkl', 'wb') as file:
            file.write(serialized_data)

        serialized_data = pickle.dumps(evaluateModelTest.Times)
        with open(output_saving_path / f'Times_{metric}_min.pkl', 'wb') as file:
            file.write(serialized_data)

        if evaluateModelTest.Attns_Mean is not None:
            serialized_data = pickle.dumps(evaluateModelTest.Attns_Mean)
            with open(output_saving_path / f'Attns_{metric}_mean.pkl', 'wb') as file:
                file.write(serialized_data)

        if evaluateModelTest.Attns_Max is not None:
            serialized_data = pickle.dumps(evaluateModelTest.Attns_Max)
            with open(output_saving_path / f'Attns_{metric}_max.pkl', 'wb') as file:
                file.write(serialized_data)

    def AtTrainingEnd(self, save_metrics_types, evaluateModelCaller, madis_network, output_saving_path):
        print('Runing best model for test set')
        best_metrics = dict()
        for save_metric_type in save_metrics_types:

            evaluateModel = evaluateModelCaller()

            modelPath = os.path.join(output_saving_path, f'model_{save_metric_type.name}_min.pt')
            state_dict = torch.load(modelPath, weights_only=True)
            evaluateModel.model.load_state_dict(state_dict)

            loss, per_variable_loss = evaluateModel.call_evaluate('test', True)

            best_metrics[save_metric_type] = dict()
            best_metrics[save_metric_type]['loss'] = loss
            best_metrics[save_metric_type]['per_variable_loss'] = per_variable_loss

            self.SaveTest(evaluateModel, madis_network, save_metric_type.name, output_saving_path)

        return best_metrics


