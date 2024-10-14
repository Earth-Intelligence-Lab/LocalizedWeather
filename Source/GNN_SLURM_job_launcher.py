import os

from Settings.Settings import TemporalDatasetSplit, StationSamplingMethod, StaticNodeType, ModelType


class SlurmJob(object):
    def __init__(self, python_file, model_name, time='96:00:00', mem='32Gb', gpu='rtx8000:1', experiment_root='/shared/home/jgiezend/wind_obs_exps', **kwargs):

        self.time = time
        self.mem = mem
        self.gpu = gpu
        self.kwargs = kwargs

        self.python_file = python_file
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
        part_2 = self.python_file + self.args
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
            # f'#SBATCH --mem={self.mem}',
            # f'#SBATCH --gres=gpu:{self.gpu}',
            '#SBATCH --partition=g6sherrie',
            # '#SBATCH --cpus-per-node=1',
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

    python_file = '/shared/home/jgiezend/wind_obs_correction/GNN_arg_parser.py'
    model_name = 'MLP'

    station_sampling_method = StationSamplingMethod.none.value
    temporal_dataset_split = TemporalDatasetSplit.holdout_fixed.value
    static_node_type = StaticNodeType.none.value
    model_type = ModelType.MLP.value

    back_hrs = 48
    lr = 1e-4
    lead_hrs = [1, 2, 4, 8, 16, 24, 36, 48]
    n_neighbors_e2ms = [0]

    for n_neighbors_e2m in n_neighbors_e2ms:
        for lead_hr in lead_hrs:
            job = SlurmJob(python_file, model_name,
                           experiment_root=f'/shared/home/jgiezend/wind_obs_exps/{model_name}',
                           epochs=200,
                           lr=lr,
                           lead_hrs=lead_hr,
                           n_neighbors_e2m=n_neighbors_e2m,
                           batch_size=256,
                           station_sampling_method = station_sampling_method,
                           temporal_dataset_split = temporal_dataset_split,
                           static_node_type = static_node_type,
                           model_type = model_type,
                           shapefile_path='Shapefiles/Regions/northeastern_buffered.shp')
            job.launch()


def point_model_w_static_node():

    python_file = '/shared/home/jgiezend/wind_obs_correction/GNN_arg_parser.py'
    model_name = 'MLP_StaticNode'

    station_sampling_method = StationSamplingMethod.none.value
    temporal_dataset_split = TemporalDatasetSplit.holdout_fixed.value
    static_node_type = StaticNodeType.terrain_lclu.value
    model_type = ModelType.MLP.value

    back_hrs = 48
    lr = 1e-4
    lead_hrs = [1, 2, 4, 8, 16, 24, 36, 48]
    n_neighbors_e2ms = [0, 8]

    for n_neighbors_e2m in n_neighbors_e2ms:
        for lead_hr in lead_hrs:
            job = SlurmJob(python_file, model_name,
                           experiment_root=f'/shared/home/jgiezend/wind_obs_exps/{model_name}',
                           epochs=200,
                           lr=lr,
                           lead_hrs=lead_hr,
                           n_neighbors_e2m=n_neighbors_e2m,
                           batch_size=256,
                           station_sampling_method = station_sampling_method,
                           temporal_dataset_split = temporal_dataset_split,
                           static_node_type = static_node_type,
                           model_type = model_type,
                           shapefile_path='Shapefiles/Regions/northeastern_buffered.shp')
            job.launch()

def graph_model():

    python_file = '/shared/home/jgiezend/wind_obs_correction/GNN_arg_parser.py'
    model_name = 'GNN_Tanh'

    station_sampling_method = StationSamplingMethod.none.value
    temporal_dataset_split = TemporalDatasetSplit.holdout_fixed.value
    static_node_type = StaticNodeType.none.value

    back_hrs = 48
    lr = 1e-4
    lead_hrs = [1, 2, 4, 8, 16, 24, 36, 48]
    n_neighbors_e2ms = [0, 8]

    for n_neighbors_e2m in n_neighbors_e2ms:
        for lead_hr in lead_hrs:
            job = SlurmJob(python_file, model_name,
                           experiment_root=f'/shared/home/jgiezend/wind_obs_exps/{model_name}',
                           epochs=200,
                           lr=lr,
                           lead_hrs=lead_hr,
                           n_neighbors_e2m=n_neighbors_e2m,
                           batch_size=128,
                           station_sampling_method = station_sampling_method,
                           temporal_dataset_split = temporal_dataset_split,
                           static_node_type = static_node_type,
                           shapefile_path='Shapefiles/Regions/northeastern_buffered.shp')
            job.launch()
def graph_model_w_static_nodes():

    python_file = '/shared/home/jgiezend/wind_obs_correction/GNN_arg_parser.py'
    model_name = 'GNN_Correction_StaticNodes'

    station_sampling_method = StationSamplingMethod.none.value
    temporal_dataset_split = TemporalDatasetSplit.holdout_fixed.value
    static_node_type = StaticNodeType.terrain_lclu.value
    model_type = ModelType.GNN.value

    back_hrs = 48
    lr = 1e-4
    lead_hrs = [1, 2, 4, 8, 16, 24, 36, 48]
    n_neighbors_e2ms = [0, 8]

    for n_neighbors_e2m in n_neighbors_e2ms:
        for lead_hr in lead_hrs:
            job = SlurmJob(python_file, model_name,
                           experiment_root=f'/shared/home/jgiezend/wind_obs_exps/{model_name}',
                           epochs=200,
                           lr=lr,
                           lead_hrs=lead_hr,
                           n_neighbors_e2m=n_neighbors_e2m,
                           batch_size=256,
                           station_sampling_method = station_sampling_method,
                           temporal_dataset_split = temporal_dataset_split,
                           static_node_type = static_node_type,
                           model_type = model_type,
                           shapefile_path='Shapefiles/Regions/northeastern_buffered.shp')
            job.launch()


def graph_model_arbitrary_lh0():

    python_file = '/shared/home/jgiezend/wind_obs_correction/GNN_arg_parser.py'
    model_name = 'GNN_Arbitrary_lh0'

    station_sampling_method = StationSamplingMethod.random_hold_out.value
    temporal_dataset_split = TemporalDatasetSplit.none.value
    static_node_types = [StaticNodeType.terrain_lclu.value]

    back_hrs = 48
    lr = 1e-4
    lead_hrs = [0]
    n_neighbors_e2ms = [0]

    for static_node_type in static_node_types:
        for n_neighbors_e2m in n_neighbors_e2ms:
            for lead_hr in lead_hrs:
                job = SlurmJob(python_file, model_name,
                               experiment_root=f'/shared/home/jgiezend/wind_obs_exps/{model_name}',
                               epochs=200,
                               lr=lr,
                               lead_hrs=lead_hr,
                               n_neighbors_e2m=n_neighbors_e2m,
                               batch_size=128,
                               station_sampling_method = station_sampling_method,
                               temporal_dataset_split = temporal_dataset_split,
                               static_node_type = static_node_type,
                               shapefile_path='Shapefiles/Regions/northeastern_buffered.shp')
                job.launch()


def graph_model_arbitrary():

    python_file = '/shared/home/jgiezend/wind_obs_correction/GNN_arg_parser.py'
    model_name = 'GNN_Arbitrary'

    station_sampling_method = StationSamplingMethod.random_hold_out.value
    temporal_dataset_split = TemporalDatasetSplit.none.value
    static_node_types = [StaticNodeType.terrain_lclu.value]

    back_hrs = 48
    lr = 1e-4
    lead_hrs = [1, 2, 4, 8, 16, 24, 36, 48]
    n_neighbors_e2ms = [8]

    for static_node_type in static_node_types:
        for n_neighbors_e2m in n_neighbors_e2ms:
            for lead_hr in lead_hrs:
                job = SlurmJob(python_file, model_name,
                               experiment_root=f'/shared/home/jgiezend/wind_obs_exps/{model_name}',
                               epochs=200,
                               lr=lr,
                               lead_hrs=lead_hr,
                               n_neighbors_e2m=n_neighbors_e2m,
                               batch_size=256,
                               station_sampling_method = station_sampling_method,
                               temporal_dataset_split = temporal_dataset_split,
                               static_node_type = static_node_type,
                               shapefile_path='Shapefiles/Regions/northeastern_buffered.shp')
                job.launch()


if __name__ == '__main__':
    # graph_model()
    point_model()
    # graph_model_arbitrary_lh0()
    # graph_model_arbitrary()

    point_model_w_static_node()
    graph_model_w_static_nodes()
