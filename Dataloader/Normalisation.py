import torch


class Normalisation:
    def __init__(self, datalist, var_name, how='minmax'):
        if how=='minmax':
            min = torch.min(torch.array([data[var_name].min for data in datalist]).flatten())
            max = torch.max(torch.array([data[var_name].max for data in datalist]).flatten())
            self.normaliser = MaxMinNormalizer(min, max)
            return

        if how=='standardisation':
            means = [data[var_name].mean for data in datalist]
            stds = [data[var_name].std for data in datalist]
            ns = [data[var_name].n for data in datalist]
            return