# Author: Qidong Yang & Jonathan Giezendanner

from enum import Enum


class ModelType(Enum):
    GNN = 1
    ViT = 2


class InterpolationType(Enum):
    none = -1
    Nearest = 0
    BiCubic = 1
    Stacked = 2


class NetworkConstructionMethod(Enum):
    none = 0
    KNN = 1
    DELAUNAY = 2
    FULLY_CONNECTED = 3


class LossFunctionType(Enum):
    MSE = 0
    WIND_VECTOR = 1
    CUSTOM = 2
    MAE = 3


class EnvVariables(Enum):
    u = 0
    v = 1
    temp = 2
    dewpoint = 3
    solar_radiation = 4


class ScalingType(Enum):
    MinMax = 0
    Standard = 1
