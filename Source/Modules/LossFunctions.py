import torch


def wind_loss(output, target):
    u_error = torch.pow(output[:, :, 0] - target[:, :, 0], 2)
    v_error = torch.pow(output[:, :, 1] - target[:, :, 1], 2)

    error = torch.sqrt(u_error + v_error)

    loss = torch.mean(error)

    return loss