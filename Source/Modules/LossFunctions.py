import torch


def wind_loss(output, target):
    u_error = output[..., 0] - target[..., 0]
    v_error = output[..., 1] - target[..., 1]

    error = torch.sqrt((u_error**2) + (v_error**2))

    loss = torch.mean(error)

    return loss