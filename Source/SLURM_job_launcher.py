# Author: Qidong Yang & Jonathan Giezendanner

import os

from Settings.Settings import *

##### settings
NB_GPUS = 1
PATH_TO_PYTHON_FILE = 'Source/Arg_Parser.py'
PATH_TO_DATA = 'WindDataNE-US/'
PATH_TO_OUTPUT_FOLDER = 'wind_obs_exps/'
WALL_TIME = '96:00:00'

MODULE_ACTIVATION = [
    'echo "starting job $SLURM_JOB_ID"',
    'source deeplearning/bin/activate',
]
SLURM_ARGS = [
    '#SBATCH --cpus-per-task=16',
    '#SBATCH --mem=64Gb',
    # f'#SBATCH --gres=gpu:rtx8000:1',
    f'#SBATCH --nodes=1',
    f'#SBATCH --ntasks-per-node=1',
    # '#SBATCH --partition=cpu-gpu-v100',
    '#SBATCH --partition=cpu-gpu-rtx8000',
    f'#SBATCH --gpus={NB_GPUS}',
    # '#SBATCH --gpus-per-node=1',
]


class SlurmJob(object):
    def __init__(self,
                 model_name,
                 python_file=PATH_TO_PYTHON_FILE,
                 data_path=PATH_TO_DATA, experiment_root=PATH_TO_OUTPUT_FOLDER, time=WALL_TIME, **kwargs):

        self.time = time
        self.kwargs = kwargs

        self.python_file = python_file
        self.data_path = data_path
        self.job_name = model_name + ''.join([f'--{k}={v}' for k, v in self.kwargs.items() if
                                              k not in ['interpolation_type', 'back_hrs', 'n_years', 'batch_size',
                                                        'hidden_dim', 'arbitrary_location', 'shapefile_path',
                                                        'model_type', 'madis_vars_o', 'madis_vars_i', 'external_vars',
                                                        'loss_function_type']])
        self.job_name = self.job_name.replace('(', '_')
        self.job_name = self.job_name.replace(')', '_')
        self.job_name = self.job_name.replace('[', '_')
        self.job_name = self.job_name.replace(']', '_')
        self.job_name = self.job_name.replace(' ', '_')
        self.job_name = self.job_name.replace(',', '_')
        self.job_name = self.job_name.replace('/', '_')
        self.job_name = self.job_name.replace('.', '_')

        self.experiment_path = experiment_root + '/' + model_name + '/' + self.job_name + '/'

        self.output_path = self.experiment_path + 'outputs' + '/'
        self.slurm_report_path = self.experiment_path + 'slurm_reports' + '/'
        self.slurm_code_path = self.experiment_path + 'slurm_codes' + '/'

        self.slurm_filename = 'slurm_script.sh'
        self.slurm_output_filename = 'slurm_output.txt'
        self.slurm_error_filename = 'slurm_error.txt'

    @property
    def args(self):

        args = []
        for k, v in self.kwargs.items():
            if isinstance(v, tuple):
                arg_str = f'--{k} ' + ' '.join([str(i) for i in v])
            else:
                arg_str = f'--{k} {v}'

            args.append(arg_str)

        if len(args) > 0:

            return ' ' + ' '.join(args)
        else:

            return ''

    @property
    def command(self):

        part_1 = 'python '
        part_2 = self.python_file + self.args + f' --data_path={self.data_path}'
        part_3 = ' ' + f'--output_saving_path={self.output_path}'

        return part_1 + part_2 + part_3

    @property
    def setup(self):

        lines = MODULE_ACTIVATION

        return lines

    @property
    def lines(self):

        lines = [
            '#!/bin/bash',
            f'#SBATCH --job-name={self.job_name}',
            f'#SBATCH --output={self.slurm_report_path + self.slurm_output_filename}',
            f'#SBATCH --error={self.slurm_report_path + self.slurm_error_filename}',
            '#SBATCH --ntasks=1',
            f'#SBATCH --time={self.time}'
        ]
        lines.extend(SLURM_ARGS)

        lines = lines + [''] + self.setup + ['', self.command]

        return lines

    @property
    def text(self):

        return '\n'.join(self.lines)

    def launch(self):

        os.system(f'mkdir -p {self.output_path}')
        os.system(f'mkdir -p {self.slurm_report_path}')
        os.system(f'mkdir -p {self.slurm_code_path}')

        with open(os.path.join(self.slurm_code_path, self.slurm_filename), 'w') as f:
            f.write(self.text)

        os.system(f'cat {os.path.join(self.slurm_code_path, self.slurm_filename)} | sbatch')


def transformer_hrrr_all_vars_model():
    model_name = 'Transformer_all_vars_HRRR'
    model_type = ModelType.ViT.value

    Lead_hrs = [1, 2, 4, 8, 12, 18, 24, 36, 48]

    for reanalysis in [0, 1]:
        for lead_hrs in Lead_hrs:
            job = SlurmJob(model_name,
                           lead_hrs=lead_hrs,
                           n_neighbors_e2m=0,
                           n_neighbors_h2m=1,
                           n_neighbors_m2m=0,
                           interpolation_type=InterpolationType.Nearest.value,
                           hrrr_analysis_only=reanalysis,
                           model_type=model_type,
                           madis_vars_i=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                         EnvVariables.dewpoint.value),
                           madis_vars_o=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                         EnvVariables.dewpoint.value),
                           external_vars=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                          EnvVariables.dewpoint.value),
                           loss_function_type=LossFunctionType.CUSTOM.value,
                           shapefile_path='Shapefiles/Regions/northeastern_buffered.shp')

            job.launch()


def transformer_era5_all_vars_model():
    model_name = 'Transformer_all_vars_ERA5'
    model_type = ModelType.ViT.value

    Lead_hrs = [1, 2, 4, 8, 12, 18, 24, 36, 48]

    for lead_hrs in Lead_hrs:
        job = SlurmJob(model_name,
                       lead_hrs=lead_hrs,
                       n_neighbors_e2m=1,
                       n_neighbors_h2m=0,
                       n_neighbors_m2m=0,
                       interpolation_type=InterpolationType.Nearest.value,
                       model_type=model_type,
                       madis_vars_i=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                     EnvVariables.dewpoint.value),
                       madis_vars_o=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                     EnvVariables.dewpoint.value),
                       external_vars=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                      EnvVariables.dewpoint.value),
                       loss_function_type=LossFunctionType.CUSTOM.value,
                       shapefile_path='Shapefiles/Regions/northeastern_buffered.shp')

        job.launch()


def transformer_all_vars_model():
    model_name = 'Transformer_all_vars'
    model_type = ModelType.ViT.value

    Lead_hrs = [1, 2, 4, 8, 12, 18, 24, 36, 48]

    for lead_hrs in Lead_hrs:
        job = SlurmJob(model_name,
                       lead_hrs=lead_hrs,
                       n_neighbors_e2m=0,
                       n_neighbors_h2m=0,
                       n_neighbors_m2m=0,
                       model_type=model_type,
                       madis_vars_i=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                     EnvVariables.dewpoint.value),
                       madis_vars_o=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                     EnvVariables.dewpoint.value),
                       external_vars=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                      EnvVariables.dewpoint.value),
                       loss_function_type=LossFunctionType.CUSTOM.value,
                       shapefile_path='Shapefiles/Regions/northeastern_buffered.shp')

        job.launch()


################## MLPPNN #################

def MLPMPNN_hrrr_all_vars_model():
    model_name = 'MLPMPNN_all_vars_HRRR'
    model_type = ModelType.GNN.value

    Lead_hrs = [1, 2, 4, 8, 12, 18, 24, 36, 48]

    for reanalysis in [0, 1]:
        for lead_hrs in Lead_hrs:
            job = SlurmJob(model_name,
                           lead_hrs=lead_hrs,
                           n_neighbors_e2m=0,
                           n_neighbors_h2m=1,
                           n_neighbors_m2m=0,
                           hrrr_analysis_only=reanalysis,
                           model_type=model_type,
                           n_passing=0,
                           madis_vars_i=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                         EnvVariables.dewpoint.value),
                           madis_vars_o=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                         EnvVariables.dewpoint.value),
                           external_vars=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                          EnvVariables.dewpoint.value),
                           loss_function_type=LossFunctionType.CUSTOM.value,
                           shapefile_path='Shapefiles/Regions/northeastern_buffered.shp')

            job.launch()


def MLPMPNN_era5_all_vars_model():
    model_name = 'MLPMPNN_all_vars_ERA5'
    model_type = ModelType.GNN.value

    Lead_hrs = [1, 2, 4, 8, 12, 18, 24, 36, 48]

    for lead_hrs in Lead_hrs:
        job = SlurmJob(model_name,
                       lead_hrs=lead_hrs,
                       n_neighbors_e2m=1,
                       n_neighbors_h2m=0,
                       n_neighbors_m2m=0,
                       model_type=model_type,
                       n_passing=0,
                       madis_vars_i=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                     EnvVariables.dewpoint.value),
                       madis_vars_o=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                     EnvVariables.dewpoint.value),
                       external_vars=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                      EnvVariables.dewpoint.value),
                       loss_function_type=LossFunctionType.CUSTOM.value,
                       shapefile_path='Shapefiles/Regions/northeastern_buffered.shp')

        job.launch()


def MLPMPNN_all_vars_model():
    model_name = 'MLPMPNN_all_vars'
    model_type = ModelType.GNN.value

    Lead_hrs = [1, 2, 4, 8, 12, 18, 24, 36, 48]

    for lead_hrs in Lead_hrs:
        job = SlurmJob(model_name,
                       lead_hrs=lead_hrs,
                       n_neighbors_e2m=0,
                       n_neighbors_h2m=0,
                       n_neighbors_m2m=0,
                       n_passing=0,
                       model_type=model_type,
                       madis_vars_i=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                     EnvVariables.dewpoint.value),
                       madis_vars_o=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                     EnvVariables.dewpoint.value),
                       external_vars=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                      EnvVariables.dewpoint.value),
                       loss_function_type=LossFunctionType.CUSTOM.value,
                       shapefile_path='Shapefiles/Regions/northeastern_buffered.shp')

        job.launch()


################## GNN #################


def GNN_hrrr_all_vars_model():
    model_name = 'GNN_all_vars_HRRR'
    model_type = ModelType.GNN.value

    Lead_hrs = [1, 2, 4, 8, 12, 18, 24, 36, 48]

    for reanalysis in [0, 1]:
        for lead_hrs in Lead_hrs:
            job = SlurmJob(model_name,
                           lead_hrs=lead_hrs,
                           n_neighbors_e2m=0,
                           n_neighbors_h2m=8,
                           hrrr_analysis_only=reanalysis,
                           model_type=model_type,
                           madis_vars_i=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                         EnvVariables.dewpoint.value),
                           madis_vars_o=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                         EnvVariables.dewpoint.value),
                           external_vars=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                          EnvVariables.dewpoint.value),
                           loss_function_type=LossFunctionType.CUSTOM.value,
                           shapefile_path='Shapefiles/Regions/northeastern_buffered.shp')

            job.launch()


def GNN_era5_all_vars_model():
    model_name = 'GNN_all_vars_ERA5'
    model_type = ModelType.GNN.value

    Lead_hrs = [1, 2, 4, 8, 12, 18, 24, 36, 48]

    for lead_hrs in Lead_hrs:
        job = SlurmJob(model_name,
                       lead_hrs=lead_hrs,
                       n_neighbors_e2m=8,
                       n_neighbors_h2m=0,
                       model_type=model_type,
                       madis_vars_i=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                     EnvVariables.dewpoint.value),
                       madis_vars_o=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                     EnvVariables.dewpoint.value),
                       external_vars=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                      EnvVariables.dewpoint.value),
                       loss_function_type=LossFunctionType.CUSTOM.value,
                       shapefile_path='Shapefiles/Regions/northeastern_buffered.shp')

        job.launch()


def GNN_all_vars_model():
    model_name = 'GNN_all_vars'
    model_type = ModelType.GNN.value

    Lead_hrs = [1, 2, 4, 8, 12, 18, 24, 36, 48]

    for lead_hrs in Lead_hrs:
        job = SlurmJob(model_name,
                       lead_hrs=lead_hrs,
                       n_neighbors_e2m=0,
                       n_neighbors_h2m=0,
                       model_type=model_type,
                       madis_vars_i=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                     EnvVariables.dewpoint.value),
                       madis_vars_o=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                     EnvVariables.dewpoint.value),
                       external_vars=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                      EnvVariables.dewpoint.value),
                       loss_function_type=LossFunctionType.CUSTOM.value,
                       shapefile_path='Shapefiles/Regions/northeastern_buffered.shp')

        job.launch()


if __name__ == '__main__':
    transformer_hrrr_all_vars_model()
    transformer_era5_all_vars_model()
    transformer_all_vars_model()
    MLPMPNN_hrrr_all_vars_model()
    MLPMPNN_era5_all_vars_model()
    MLPMPNN_all_vars_model()
    GNN_hrrr_all_vars_model()
    GNN_era5_all_vars_model()
    GNN_all_vars_model()
