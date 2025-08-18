# Author: Qidong Yang & Jonathan Giezendanner

import json
from pathlib import Path

import Main
from PostProcessInputs import PostProcessArgs
from Settings.Settings import *


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


if __name__ == '__main__':
    args = AttrDict()

    model_name = 'Transformer'

    args.lead_hrs = 1
    args.n_neighbors_e2m = 1
    args.n_neighbors_h2m = 0
    args.hrrr_analysis_only = False
    args.interpolation_type = InterpolationType.Nearest.value

    args.output_saving_path = f'ModelOutputs/{model_name}'

    ######### args after this are not included in path name

    args.coords = None
    args.shapefile_path = 'Shapefiles/Regions/northeastern_buffered.shp'
    args.back_hrs = 48
    args.hidden_dim = 128
    args.lr = 1e-4
    args.epochs = 20
    args.batch_size = 128
    args.weight_decay = 1e-4
    args.model_type = ModelType.GNN.value
    args.network_construction_method = NetworkConstructionMethod.FULLY_CONNECTED.value
    args.n_years = 5
    args.madis_control_ratio = .9
    args.n_passing = 4
    args.n_neighbors_m2m = 0
    args.loss_function_type = LossFunctionType.CUSTOM.value

    args.madis_vars_i = [EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                         EnvVariables.dewpoint.value]
    args.madis_vars_o = [EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                         EnvVariables.dewpoint.value]

    args.external_vars = [EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value]

    args.data_path = '/Users/jonathangiezendanner/Documents/MIT/Projects/WindDataNE-US/'
    args.show_progress_bar = True

    save_args = args.copy()

    for k in save_args.keys():
        v = save_args[k]
        if issubclass(type(v), Enum):
            save_args[k] = v.value

    outputPath = (Path(args.data_path) / args.output_saving_path / 'params.json')

    outputPath.parent.mkdir(exist_ok=True, parents=True)

    with open(outputPath, 'w') as f:
        json.dump(save_args, f)

    args = PostProcessArgs(args)

    args.use_wb = False

    Main.Main(args).Run()
