# Author: Qidong Yang & Jonathan Giezendanner

import argparse
import json
from pathlib import Path

import Main
from PostProcessInputs import PostProcessArgs
from Settings.Settings import *

if __name__ == '__main__':

    #### Configuration #####
    parser = argparse.ArgumentParser()
    ## Model
    parser.add_argument('--model_type', default=ModelType.ViT.value, type=int)

    parser.add_argument('--madis_vars_i', default=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                                   EnvVariables.dewpoint.value), type=int, nargs='+')
    parser.add_argument('--madis_vars_o', default=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                                   EnvVariables.dewpoint.value), type=int, nargs='+')

    parser.add_argument('--external_vars', default=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                                    EnvVariables.dewpoint.value), type=int, nargs='+')

    ## Graph
    parser.add_argument('--network_construction_method', default=NetworkConstructionMethod.none.value, type=int)

    ## Dataset
    parser.add_argument('--coords', default=(-74, -70, 41, 43), type=float, nargs='+')
    parser.add_argument('--shapefile_path', default=None, type=str)
    parser.add_argument('--madis_control_ratio', default=.9, type=float)

    # Experiment Hyperparameters
    parser.add_argument('--lead_hrs', default=12, type=int)
    parser.add_argument('--back_hrs', default=48, type=int)
    parser.add_argument('--n_neighbors_m2m', default=4, type=int)
    parser.add_argument('--n_neighbors_e2m', default=8, type=int)
    parser.add_argument('--n_neighbors_h2m', default=0, type=int)
    parser.add_argument('--hrrr_analysis_only', default=0, type=int)
    parser.add_argument('--n_years', default=5, type=int)
    parser.add_argument('--interpolation_type', default=InterpolationType.none.value, type=int)

    # Model Hyperparameters
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--loss_function_type', default=LossFunctionType.CUSTOM.value, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--n_passing', default=4, type=int)

    # code setup
    parser.add_argument('--show_progress_bar', default=False, type=bool)

    # file systems
    parser.add_argument('--data_path', default='/shared/home/jgiezend/data/WindData/', type=str)
    parser.add_argument('--output_saving_path', default='tmp', type=str)

    # To use weights and biases (wb), you need to modify Utils/Telemetry.py and specify `project` and `entity`
    parser.add_argument('--use_wb', default=0, type=int)

    ## process args
    args = parser.parse_args()

    save_args = vars(args).copy()

    for k in save_args.keys():
        v = save_args[k]
        if issubclass(type(v), Enum):
            save_args[k] = v.value

    # save config to json
    outputPath = (Path(args.output_saving_path) / 'params.json')

    with open(outputPath, 'w') as f:
        json.dump(save_args, f)

    args = PostProcessArgs(args)

    args.use_wb = args.use_wb == 1

    # run main code
    Main.Main(args).Run()
