from enum import Enum


class ModelType(Enum):
    MLP = 0
    GNN = 1
    MPNN_MLP = 2

class MLPERA5InterpolationType(Enum):
    Nearest = 0
    Linear = 1

class NetworkConstructionMethod(Enum):
    KNN = 1
    DELAUNAY = 2

class LossFunctionType(Enum):
    MSE = 0
    WIND_VECTOR = 1
