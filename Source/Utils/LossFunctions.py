# Author: Qidong Yang & Jonathan Giezendanner

import numpy as np
import torch
from torch import nn

from Settings.Settings import LossFunctionType, EnvVariables


def wind_loss(output, target, logical=None, reduction='mean'):
    u_error = output[..., 0] - target[..., 0]
    v_error = output[..., 1] - target[..., 1]

    error = torch.sqrt((u_error**2) + (v_error**2) + torch.finfo(torch.float32).eps)

    if logical is not None:
        error = error[logical[..., 0]]

    if reduction == 'mean':
        return torch.mean(error)
    elif reduction == 'sum':
        return torch.sum(error)
    else:
        print('Invalid reduction type')

    return None

def custom_loss(output, target, u_index, v_index, other_index, logical=None, reduction='mean'):
    error = (output - target)**2

    u_error = error[..., u_index]
    v_error = error[..., v_index]
    wind_error = torch.sqrt(u_error + v_error + torch.finfo(torch.float32).eps).unsqueeze(-1)

    error = torch.sqrt(error[..., other_index] + torch.finfo(torch.float32).eps)

    if logical is not None:
        wind_error = wind_error[logical[..., 0]].flatten()
        error = error[logical[..., other_index]].flatten()

    error = torch.cat((wind_error, error), dim=-1)

    if reduction == 'mean':
        return torch.mean(error)
    elif reduction == 'sum':
        return torch.sum(error)
    else:
        print('Invalid reduction type')

    return None

def GetLossFunction(loss_function_type, madis_vars_o):
    if loss_function_type == LossFunctionType.MSE:
        loss_function = lambda output, target, logical: nn.MSELoss(reduction='mean')(output, target)
    elif loss_function_type == LossFunctionType.WIND_VECTOR:
        loss_function = lambda output, target, logical: wind_loss(output, target)
    elif loss_function_type == LossFunctionType.CUSTOM:
        u_index = np.array(madis_vars_o) == EnvVariables.u
        v_index = np.array(madis_vars_o) == EnvVariables.v
        other_index = ~(u_index | v_index)
        u_index = np.where(u_index)[0][0]
        v_index = np.where(v_index)[0][0]
        other_index = np.where(other_index)[0]

        loss_function = lambda output, target, logical: custom_loss(output, target, u_index, v_index, other_index)
    else:
        raise ValueError(f"Unknown loss function type: {loss_function_type}")

    return loss_function

def GetLossFunctionReport(loss_function_type, madis_vars_o):
    if loss_function_type == LossFunctionType.MSE:
        loss_function_report = nn.MSELoss(reduction='sum')
    elif loss_function_type == LossFunctionType.WIND_VECTOR:
        loss_function_report = lambda output, target: wind_loss(output, target, reduction='sum')
    elif loss_function_type == LossFunctionType.CUSTOM:
        u_index = np.array(madis_vars_o) == EnvVariables.u
        v_index = np.array(madis_vars_o) == EnvVariables.v
        other_index = ~(u_index | v_index)
        u_index = np.where(u_index)[0][0]
        v_index = np.where(v_index)[0][0]
        other_index = np.where(other_index)[0]

        loss_function_report = lambda output, target: custom_loss(output, target, u_index, v_index, other_index,
                                                                  reduction='sum')
    else:
        raise ValueError(f"Unknown loss function type: {loss_function_type}")

    return loss_function_report

def SetupSaveMetrics(save_metrics_types, madis_vars_o):
    save_metrics = dict()
    for save_metric_type in save_metrics_types:
        if save_metric_type == LossFunctionType.MSE:
            save_metrics[save_metric_type] = lambda output, target, logical: nn.MSELoss(reduction='sum')(output, target)
            continue

        if save_metric_type == LossFunctionType.WIND_VECTOR:
            save_metrics[save_metric_type] = lambda output, target, logical: wind_loss(output, target, reduction='sum')
            continue

        if save_metric_type == LossFunctionType.CUSTOM:
            u_index = np.array(madis_vars_o) == EnvVariables.u
            v_index = np.array(madis_vars_o) == EnvVariables.v
            other_index = ~(u_index | v_index)
            u_index = np.where(u_index)[0][0]
            v_index = np.where(v_index)[0][0]
            other_index = np.where(other_index)[0]

            save_metrics[save_metric_type] = lambda output, target, logical: custom_loss(output, target, u_index, v_index,
                                                                                     other_index, reduction='sum')
            continue

    return save_metrics