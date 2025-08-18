import os


class SlurmJob(object):
    def __init__(self, python_file, task_name, time='1:00:00', mem='10Gb',
                 experiment_root='/home/mila/q/qidong.yang/scratch/HRRR/docs/', **kwargs):

        self.time = time
        self.mem = mem
        self.kwargs = kwargs

        self.python_file = python_file
        self.job_name = task_name + ''.join([f'--{k}={v}' for k, v in self.kwargs.items() if k not in ['output_dir']])

        self.experiment_path = experiment_root + self.job_name + '/'
        # self.output_path = self.experiment_path + 'outputs'
        self.slurm_report_path = self.experiment_path + 'slurm_reports' + '/'
        self.slurm_code_path = self.experiment_path + 'slurm_codes' + '/'

        self.slurm_filename = 'slurm_script.sh'
        self.slurm_output_filename = 'slurm_output.txt'
        self.slurm_error_filename = 'slurm_error.txt'

    @property
    def args(self):

        args = []
        for k, v in self.kwargs.items():
            args.append(f'--{k}={v}')

        if len(args) > 0:

            return ' ' + ' '.join(args)
        else:

            return ''

    @property
    def command(self):

        part_1 = 'python '
        part_2 = self.python_file + self.args

        return part_1 + part_2

    @property
    def setup(self):

        lines = [
            'module load anaconda/3',
            ' ',
            'conda activate forecast_download',
            ' ',
            self.command,
        ]

        return lines

    @property
    def lines(self):

        lines = [
            '#!/bin/bash',
            f'#SBATCH --job-name={self.job_name}',
            f'#SBATCH --output={self.slurm_report_path + self.slurm_output_filename}',
            '#SBATCH --ntasks=1',
            '#SBATCH --cpus-per-task=2',
            f'#SBATCH --time={self.time}',
            f'#SBATCH --mem={self.mem}',
            '#SBATCH --partition=long',
            ' ',
        ]

        lines = lines + self.setup

        return lines

    @property
    def text(self):

        return '\n'.join(self.lines)

    def launch(self):

        # os.system(f'mkdir -p {self.output_path}')
        os.system(f'mkdir -p {self.slurm_report_path}')
        os.system(f'mkdir -p {self.slurm_code_path}')

        with open(os.path.join(self.slurm_code_path, self.slurm_filename), 'w') as f:
            f.write(self.text)

        os.system(f'cat {os.path.join(self.slurm_code_path, self.slurm_filename)} | sbatch')


def download():
    python_file = '/home/mila/q/qidong.yang/wind_obs_correction/download_code/madis/download_beam.py'
    task_name = 'HRRR_download'
    output_dir = '/home/mila/q/qidong.yang/scratch/HRRR/'
    years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]

    for year in years:
        for month in range(1, 13):
            job = SlurmJob(python_file, task_name, year=year, month=month, output_dir=output_dir)
            job.launch()


if __name__ == '__main__':
    download()
