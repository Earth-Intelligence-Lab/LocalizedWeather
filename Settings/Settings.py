from enum import Enum


class NetworkConstructionMethod(Enum):
    KNN = 1


class StationSamplingMethod(Enum):
    none = 0 # no holdout network
    random_shared = 1 # same known and unknown for everyone
    random_hold_out = 2 # train known / uknkown, test and val get train known/unknown and other unknown


class TemporalDatasetSplit(Enum):
    none = 0 # no temporal split
    holdout_fixed = 1 # 3 years for train, 1 for test, 1 for val

class StaticNodeType(Enum):
    none = 0
    terrain = 1
    lclu = 2
    terrain_lclu = 3

class ModelType(Enum):
    MLP = 0
    GNN = 1
