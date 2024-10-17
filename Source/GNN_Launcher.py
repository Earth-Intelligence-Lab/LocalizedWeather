# Author: Qidong Yang & Jonathan Giezendanner

import json
from pathlib import Path

from Settings.Settings import *
from Source import GNN_Main


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


args = AttrDict()

model_name = 'MPNN'

args.lead_hrs = 4
args.n_neighbors_e2m = 8

args.output_saving_path = f'ModelOutputs/{model_name}_' + ''.join(
    [f'_{k}={v.name if issubclass(type(v), Enum) else v}' for k, v in args.items()])

######### args after this are not included in path name

args.coords = None
args.shapefile_path = 'Shapefiles/Regions/northeastern_buffered.shp'
args.back_hrs = 48
args.hidden_dim = 128
args.lr = 1e-4
args.epochs = 10
args.batch_size = 128
args.weight_decay = 1e-4
args.model_type = ModelType.GNN
args.n_years = 5
args.madis_control_ratio = .9
args.n_passing = 4
args.n_neighbors_m2m = 5

save_args = args.copy()

for k in save_args.keys():
    v = save_args[k]
    if issubclass(type(v), Enum):
        save_args[k] = v.name

args.eval_interval = 5
args.root_path = '/Users/jonathangiezendanner/Documents/MIT/Projects/WindDataNE-US/'
args.show_progress_bar = True

outputPath = (Path(args.root_path).parent / args.output_saving_path / 'params.json')

outputPath.parent.mkdir(exist_ok=True, parents=True)

with open(outputPath, 'w') as f:
    json.dump(save_args, f)

GNN_Main.Run(args)
