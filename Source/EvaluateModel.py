# Author: Qidong Yang & Jonathan Giezendanner

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from Settings.Settings import ModelType


class EvaluateModel:

    def __init__(self, model, data_loaders, madis_norm_dict, external_norm_dict, device, lead_hrs, madis_vars_i,
                 madis_vars_o, madis_vars, external_vars, loss_function=None, loss_function_report=None,
                 save_metrics_types=None, save_metrics_functions=None, per_variable_metrics_types=None,
                 per_variable_metrics=None, model_type=ModelType.GNN, show_progress_bar=False, optimizer=None):
        self.model = model
        self.data_loaders = data_loaders
        self.madis_norm_dict = madis_norm_dict
        self.external_norm_dict = external_norm_dict
        self.device = device
        self.lead_hrs = lead_hrs
        self.madis_vars_i = madis_vars_i
        self.madis_vars_o = madis_vars_o
        self.madis_vars = madis_vars
        self.external_vars = external_vars
        self.loss_function = loss_function
        self.loss_function_report = loss_function_report
        self.save_metrics_types = save_metrics_types
        self.save_metrics_functions = save_metrics_functions
        self.per_variable_metrics_types = per_variable_metrics_types
        self.per_variable_metrics = per_variable_metrics
        self.model_type = model_type
        self.show_progress_bar = show_progress_bar
        self.optimizer = optimizer

    def call_evaluate(self, station_type='train', save=False):
        data_loader = self.data_loaders[station_type]
        self.is_train = station_type == 'train'

        per_variable_loss = dict()

        nb_obs = dict()

        for madis_var in self.madis_vars_o:
            nb_obs[madis_var] = 0
            per_variable_loss[madis_var] = dict()
            for per_variable_metrics_type in self.per_variable_metrics_types:
                per_variable_loss[madis_var][per_variable_metrics_type] = 0

        loss_report = 0
        n_report = 0

        if save == True:
            Pred_list = []
            Target_list = []
            atten_n = None
            atten_sum = None
            atten_max = None
            time_list = []

        save_metrics_dict = dict()
        for save_metric in self.save_metrics_types:
            save_metrics_dict[save_metric] = 0

        if self.is_train:
            self.model.train()
        else:
            self.model.eval()

        print(f'runing {station_type}', flush=True)

        with torch.set_grad_enabled(self.is_train):
            loopItems = tqdm(data_loader) if self.show_progress_bar else data_loader
            for sample_k, sample in enumerate(loopItems):
                self.sample_k = sample_k
                self.sample = sample

                edge_index_m2m, madis_lat, madis_lon, madis_x, y = self.ProcessSampleMadis(self.sample)

                if self.external_norm_dict is not None:
                    edge_index_ex2m, external_lat, external_lon, external_x = self.GetERA5Sample()
                    edge_index_ex2m = edge_index_ex2m.to(self.device)
                    external_lat = external_lat.to(self.device)
                    external_lon = external_lon.to(self.device)
                    external_x = external_x.to(self.device)

                else:
                    external_lon = None
                    external_lat = None
                    external_x = None
                    edge_index_ex2m = None

                if self.is_train:
                    self.optimizer.zero_grad()

                out, alphas = self.RunModel(edge_index_ex2m, edge_index_m2m, external_lat, external_lon, external_x,
                                            madis_lat,
                                            madis_lon, madis_x)

                is_real = self.GetIsReal()
                if self.is_train:
                    ls = self.loss_function(out, y, is_real)
                    ls.backward()
                    if type(self.model) is nn.DataParallel:
                        torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), max_norm=1.0)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                loss_report += self.loss_function_report(out.detach(), y.detach()).cpu().numpy()
                n_report += out[..., 0].numel()

                for save_metric in self.save_metrics_types:
                    save_metrics_dict[save_metric] += self.save_metrics_functions[save_metric](out.detach(), y.detach(),
                                                                                               is_real).cpu().numpy()

                # denormalize output
                for k, madis_var in enumerate(self.madis_vars_o):
                    y[..., k] = self.madis_norm_dict[madis_var].decode(y[..., k])
                    out[..., k] = self.madis_norm_dict[madis_var].decode(out[..., k])
                # (n_batch, n_stations, 2)

                if save == True:
                    out_save, y_save = self.GetPredictionAndTargetForSaving(out, y)

                    if alphas is not None:
                        alphas = alphas.detach().cpu().numpy()
                        if atten_sum is None:
                            atten_sum = alphas.sum(axis=0)
                            atten_max = alphas.max(axis=0, keepdims=True)
                            atten_n = alphas.shape[0]
                        else:
                            atten_sum += alphas.sum(axis=0)
                            atten_max = np.max(np.concatenate([atten_max, alphas], axis=0), axis=0, keepdims=True)
                            atten_n += alphas.shape[0]

                    Pred_list.append(out_save)
                    Target_list.append(y_save)
                    time_list.append(self.sample['time'].numpy())

                y = y.detach()
                out = out.detach()
                for k, madis_var in enumerate(self.madis_vars_o):
                    out_k, y_k = self.GetPerVariableTargetAndPredictionFromMatrixForMonitoring(k, out, y)
                    # (n_stations, 1)
                    nb_obs[madis_var] += torch.numel(out_k)
                    for per_variable_metrics_type in self.per_variable_metrics_types:
                        per_variable_loss[madis_var][per_variable_metrics_type] += np.sum(
                            self.per_variable_metrics[per_variable_metrics_type](out_k, y_k).cpu().numpy())

        for madis_var in self.madis_vars_o:
            for per_variable_metrics_type in self.per_variable_metrics_types:
                per_variable_loss[madis_var][per_variable_metrics_type] /= nb_obs[madis_var]

        loss_report /= n_report

        if save == True:
            self.Preds = np.concatenate(Pred_list, axis=0)
            self.Targets = np.concatenate(Target_list, axis=0)
            self.Times = np.concatenate(time_list, axis=0)

            if atten_sum is not None:
                self.Attns_Mean = atten_sum / atten_n
                self.Attns_Max = atten_max.squeeze()
            else:
                self.Attns_Mean = None
                self.Attns_Max = None

        self.save_metric_dict = save_metrics_dict

        return loss_report, per_variable_loss

    def GetERA5Sample(self):
        external_lon = self.sample[f'external_lon']
        external_lat = self.sample[f'external_lat']
        edge_index_ex2m = self.sample[f'ex2m_edge_index']
        external_vals_dict = dict()
        for external_var in self.external_vars:
            external_vals_dict[external_var] = self.sample['ext_' + external_var.name]
            external_vals_dict[external_var] = self.external_norm_dict[external_var].encode(
                external_vals_dict[external_var]).unsqueeze(3)
        external_x = torch.cat(list(external_vals_dict.values()), dim=-1)
        return edge_index_ex2m, external_lat, external_lon, external_x

    def GetIsReal(self):
        return np.concatenate(list(
            map(lambda var: self.sample.get(var.name + '_is_real').unsqueeze(2) == 1,
                self.madis_vars_o)), axis=-1)

    def ProcessSampleMadis(self, sample):
        madis_lon = sample[f'madis_lon'].to(self.device)
        madis_lat = sample[f'madis_lat'].to(self.device)
        edge_index_m2m = sample[f'k_edge_index'].to(self.device)
        madis_vals_dict = dict()
        for madis_var in self.madis_vars:
            madis_vals_dict[madis_var] = sample[madis_var]
            madis_vals_dict[madis_var] = self.madis_norm_dict[madis_var].encode(madis_vals_dict[madis_var]).unsqueeze(3)
        y = self.GetTarget(madis_vals_dict).to(self.device)
        madis_x = self.GetMadisInputs(madis_vals_dict).to(self.device)
        return edge_index_m2m, madis_lat, madis_lon, madis_x, y

    def GetPredictionAndTargetForSaving(self, out, y):
        out_save = out.detach().cpu().numpy()
        y_save = y.detach().cpu().numpy()
        return out_save, y_save

    def GetPerVariableTargetAndPredictionFromMatrixForMonitoring(self, k, out, y):
        out_k = out[..., k]
        y_k = y[..., k]
        return out_k, y_k

    def GetMadisInputs(self, madis_vals_dict):
        madis_x = torch.cat(list(map(madis_vals_dict.get, self.madis_vars_i)), dim=-1)
        madis_matrix_len = madis_x.shape[2]
        madis_x = madis_x[:, :, :madis_matrix_len - self.lead_hrs, :]
        return madis_x

    def GetTarget(self, madis_vals_dict):
        y = torch.cat(list(map(lambda var: madis_vals_dict.get(var)[:, :, -1, :], self.madis_vars_o)), dim=-1)
        return y

    def RunModel(self, edge_index_ex2m, edge_index_m2m, external_lat, external_lon, external_x, madis_lat, madis_lon,
                 madis_x):
        alphas = None
        if self.model_type == ModelType.ViT:
            out, alphas = self.model(madis_x, external_x, return_attn=False)
        elif self.model_type == ModelType.GNN:
            out = self.model(madis_x,
                             madis_lon,
                             madis_lat,
                             edge_index_m2m,
                             external_lon,
                             external_lat,
                             external_x,
                             edge_index_ex2m)
        else:
            raise NotImplementedError
        return out, alphas
