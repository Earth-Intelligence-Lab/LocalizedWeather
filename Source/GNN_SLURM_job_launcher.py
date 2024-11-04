# Author: Qidong Yang & Jonathan Giezendanner

import os

from Settings.Settings import ModelType


class SlurmJob(object):
    def __init__(self,
                 model_name,
                 python_file='/shared/home/jgiezend/LocalizedWeatherGNN/Source/GNN_arg_parser.py',
                 data_path='/shared/home/jgiezend/data/WindDataNE-US/', experiment_root='', time='96:00:00', **kwargs):

        self.time = time
        self.kwargs = kwargs

        self.python_file = python_file
        self.data_path = data_path
        self.job_name = model_name + ''.join([f'--{k}={v}' for k, v in self.kwargs.items()])
        self.job_name = self.job_name.replace('(', '_')
        self.job_name = self.job_name.replace(')', '_')
        self.job_name = self.job_name.replace('[', '_')
        self.job_name = self.job_name.replace(']', '_')
        self.job_name = self.job_name.replace(' ', '_')
        self.job_name = self.job_name.replace(',', '_')
        self.job_name = self.job_name.replace('/', '_')
        self.job_name = self.job_name.replace('.', '_')

        self.experiment_path = experiment_root + '/' + self.job_name + '/'

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
            args.append(f'--{k} {v}')

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

        lines = [
            # 'module unload python',
            # 'module load anaconda/3',
            'source /shared/home/jgiezend/.bashrc',
            'mamba activate wind',
        ]

        return lines

    @property
    def lines(self):

        lines = [
            '#!/bin/bash',
            f'#SBATCH --job-name={self.job_name}',
            f'#SBATCH --output={self.slurm_report_path + self.slurm_output_filename}',
            f'#SBATCH --error={self.slurm_report_path + self.slurm_error_filename}',
            '#SBATCH --ntasks=1',
            f'#SBATCH --time={self.time}',
            f'#SBATCH --nodes=1',
            f'#SBATCH --ntasks-per-node=1',
            '#SBATCH --partition=g6sherrie',
            '#SBATCH --gpus=1',
            '#SBATCH --gpus-per-node=1',
        ]

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


def point_model():
    model_name = 'MLP'

    model_type = ModelType.MLP.value

    lead_hrs = [1, 2, 4, 8, 16, 24, 36, 48]
    n_neighbors_e2ms = [1]

    for n_neighbors_e2m in n_neighbors_e2ms:
        for lead_hr in lead_hrs:
            job = SlurmJob(model_name,
                           experiment_root=f'/shared/home/jgiezend/wind_obs_exps_release/{model_name}',
                           epochs=200,
                           lead_hrs=lead_hr,
                           n_neighbors_e2m=n_neighbors_e2m,
                           model_type=model_type,
                           shapefile_path='Shapefiles/Regions/northeastern_buffered.shp')
            job.launch()


def point_modelMPNN_MLP():
    model_name = 'MPNN_MLP'

    model_type = ModelType.MPNN_MLP.value

    lead_hrs = [1, 2, 4, 8, 16, 24, 36, 48]
    n_neighbors_e2ms = [1]

    for n_neighbors_e2m in n_neighbors_e2ms:
        for lead_hr in lead_hrs:
            job = SlurmJob(model_name,
                           experiment_root=f'/shared/home/jgiezend/wind_obs_exps_release/{model_name}',
                           epochs=200,
                           lead_hrs=lead_hr,
                           n_neighbors_e2m=n_neighbors_e2m,
                           model_type=model_type,
                           shapefile_path='Shapefiles/Regions/northeastern_buffered.shp')
            job.launch()


def graph_model():
    model_name = 'GNN'

    model_type = ModelType.GNN.value

    lead_hrs = [1, 2, 4, 8, 16, 24, 36, 48]
    n_neighbors_e2ms = [0, 8]

    for n_neighbors_e2m in n_neighbors_e2ms:
        for lead_hr in lead_hrs:
            job = SlurmJob(model_name,
                           experiment_root=f'/shared/home/jgiezend/wind_obs_exps_release/{model_name}',
                           epochs=200,
                           lead_hrs=lead_hr,
                           n_neighbors_e2m=n_neighbors_e2m,
                           model_type=model_type,
                           shapefile_path='Shapefiles/Regions/northeastern_buffered.shp')
            job.launch()


if __name__ == '__main__':
    point_model()
    # graph_model()
    # point_modelMPNN_MLP()
