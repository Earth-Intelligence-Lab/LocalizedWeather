# Author: Qidong Yang & Jonathan Giezendanner

import argparse
import json
from enum import Enum
from pathlib import Path

from Settings.Settings import ModelType
from Source import GNN_Main

#### Configuration #####
parser = argparse.ArgumentParser()

## Model
parser.add_argument('--model_type', default=ModelType.GNN.value, type=int)

## Dataset
parser.add_argument('--coords', default=(-74, -70, 41, 43), type=float, nargs='+')
parser.add_argument('--shapefile_path', default=None, type=str)
parser.add_argument('--madis_control_ratio', default=.9, type=float)

# Experiment Hyperparameters
parser.add_argument('--lead_hrs', default=4, type=int)
parser.add_argument('--back_hrs', default=48, type=int)
parser.add_argument('--n_neighbors_m2m', default=5, type=int)
parser.add_argument('--n_neighbors_e2m', default=8, type=int)
parser.add_argument('--n_years', default=5, type=int)

# Model Hyperparameters
parser.add_argument('--hidden_dim', default=128, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--n_passing', default=4, type=int)

# code setup
parser.add_argument('--eval_interval', default=5, type=int)
parser.add_argument('--show_progress_bar', default=False, type=bool)

# file systems
parser.add_argument('--root_path', default='/shared/home/jgiezend/data/WindData/', type=str)
parser.add_argument('--output_saving_path', default='tmp', type=str)

## process args
args = parser.parse_args()

args.model_type = ModelType(args.model_type)
save_args = vars(args).copy()

for k in save_args.keys():
    v = save_args[k]
    if issubclass(type(v), Enum):
        save_args[k] = v.name

# save config to json
outputPath = (Path(args.output_saving_path) / 'params.json')

with open(outputPath, 'w') as f:
    json.dump(save_args, f)

# run main code
GNN_Main.Run(args)
