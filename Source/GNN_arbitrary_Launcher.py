from pathlib import Path

import GNN_Arbitrary
# from git import Repo
import json

from Settings.Settings import *


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


args = AttrDict()

model_name = 'NoArbitrary'
# repo = Repo(".")
# commitHash = repo.git.rev_parse("HEAD")

args.p_arbitrary_nodes = .2
args.p_arbitrary_nodes_valid = .1
args.p_arbitrary_nodes_test = .1
# args.p_arbitrary_nodes = 0
args.lead_hrs = 4
args.n_neighbors_e2m = 0
args.station_sampling_method = StationSamplingMethod.none
# args.station_sampling_method = StationSamplingMethod.random_shared
# args.station_sampling_method = StationSamplingMethod.random_hold_out

# args.temporal_dataset_split = TemporalDatasetSplit.none
args.temporal_dataset_split = TemporalDatasetSplit.holdout_fixed

args.static_node_type = StaticNodeType.terrain_lclu
# args.static_node_type = StaticNodeType.none

# args.output_saving_path = f'ModelOutputs/{model_name}_{commitHash}_' + ''.join([f'_{k}={v.name if issubclass(type(v), Enum) else v}' for k, v in args.items()])
args.output_saving_path = f'ModelOutputs/{model_name}_' + ''.join([f'_{k}={v.name if issubclass(type(v), Enum) else v}' for k, v in args.items()])


######### args after this are not included in path name

args.static_node_image_size = 32

args.network_construction_method = 'KNN' #NetworkConstructionMethod.KNN

args.loss_factor_interpolate = .01
args.loss_factor_extrapolate_known = .1
args.loss_factor_extrapolate_unknown = 1

args.coords = (-74, -70, 41, 43)
args.shapefile_path = None
# args.coords = None
# args.shapefile_path = 'Shapefiles/Regions/northeastern_buffered.shp'
args.back_hrs = 24
#args.ERA5_len = 25
args.hidden_dim = 128
args.lr = 1e-4
args.epochs = 10
args.batch_size = 64
args.weight_decay = 1e-4
args.model_type = ModelType.GNN
#args.shuffle_data = False
args.n_years = 5
args.madis_control_ratio = .9
args.n_passing = 4
args.n_neighbors_m2m = 5
args.edge_type = 'None'

save_args = args.copy()

for k in save_args.keys():
    v = save_args[k]
    if issubclass(type(v), Enum):
        save_args[k] = v.name

args.eval_interval = 5
args.root_path = '/Users/jonathangiezendanner/Documents/MIT/Projects/WindData/'
args.show_progress_bar = True

outputPath = (Path(args.root_path).parent/'ModelOutputs'/args.output_saving_path/'params.json')

outputPath.parent.mkdir(exist_ok=True, parents=True)

with open(outputPath, 'w') as f:
    json.dump(save_args, f)

GNN_Arbitrary.Run(args)
