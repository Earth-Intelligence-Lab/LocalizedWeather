import argparse
import json
from enum import Enum
from pathlib import Path

import GNN_Arbitrary
from Settings.Settings import StationSamplingMethod, TemporalDatasetSplit, StaticNodeType, ModelType

##### Configuration #####
parser = argparse.ArgumentParser()


parser.add_argument('--lead_hrs', default=4, type=int)
parser.add_argument('--n_neighbors_e2m', default=4, type=int)
parser.add_argument('--station_sampling_method', default=StationSamplingMethod.none.value, type=int)
parser.add_argument('--temporal_dataset_split', default=TemporalDatasetSplit.holdout_fixed.value, type=int)

parser.add_argument('--static_node_type', default=StaticNodeType.none.value, type=int)


parser.add_argument('--static_node_image_size', default=32, type=int)

parser.add_argument('--network_construction_method', default='KNN', type=str) #NetworkConstructionMethod.KNN)

parser.add_argument('--p_arbitrary_nodes', default=0.2, type=int)
parser.add_argument('--p_arbitrary_nodes_valid', default=.1, type=float)
parser.add_argument('--p_arbitrary_nodes_test', default=.1, type=float)

parser.add_argument('--loss_factor_interpolate', default=.1, type=float)
parser.add_argument('--loss_factor_extrapolate_known', default=.1, type=float)
parser.add_argument('--loss_factor_extrapolate_unknown', default=1, type=float)


parser.add_argument('--coords', default=(-74, -70, 41, 43), type=float, nargs='+')
parser.add_argument('--shapefile_path', default=None, type=str)
parser.add_argument('--back_hrs', default=48, type=int)
#parser.add_argument('--ERA5_len', default=25)
parser.add_argument('--hidden_dim', default=128, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--model_type', default=ModelType.GNN.value, type=int)
#parser.add_argument('--shuffle_data', default=False)
parser.add_argument('--n_years', default=5, type=int)
parser.add_argument('--madis_control_ratio', default=.9, type=float)
parser.add_argument('--n_passing', default=4, type=int)
parser.add_argument('--n_neighbors_m2m', default=5, type=int)
parser.add_argument('--edge_type', default='None', type=str)

parser.add_argument('--eval_interval', default=5, type=int)

parser.add_argument('--root_path', default='/shared/home/jgiezend/data/WindData/', type=str)
parser.add_argument('--output_saving_path', default='tmp', type=str)
parser.add_argument('--show_progress_bar', default=False, type=bool)

args = parser.parse_args()

args.station_sampling_method = StationSamplingMethod(args.station_sampling_method)
args.temporal_dataset_split = TemporalDatasetSplit(args.temporal_dataset_split)
args.static_node_type = StaticNodeType(args.static_node_type)
args.model_type = ModelType(args.model_type)

save_args = vars(args).copy()

for k in save_args.keys():
    v = save_args[k]
    if issubclass(type(v), Enum):
        save_args[k] = v.name

outputPath = (Path(args.output_saving_path)/'params.json')

with open(outputPath, 'w') as f:
    json.dump(save_args, f)

GNN_Arbitrary.Run(args)
