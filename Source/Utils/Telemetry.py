# Author: Qidong Yang & Jonathan Giezendanner

import numpy as np
from matplotlib import pyplot as plt
from torch import nn

from Settings.Settings import LossFunctionType


class Telemetry:
    def __init__(self, madis_vars_o, per_variable_metrics_types):
        self.madis_vars_o = madis_vars_o
        self.per_variable_metrics_types = per_variable_metrics_types

        self.per_variable_metrics = dict()
        for per_variable_metrics_type in per_variable_metrics_types:
            if per_variable_metrics_type == LossFunctionType.MSE:
                self.per_variable_metrics[per_variable_metrics_type] = nn.MSELoss(reduction='sum')
                continue

            if per_variable_metrics_type == LossFunctionType.MAE:
                self.per_variable_metrics[per_variable_metrics_type] = nn.L1Loss(reduction='sum')
                continue

        ###### Training ######
        self.losses = dict()
        self.per_variable_losses = dict()

        for dataset in ['train', 'val', 'test']:
            self.losses[dataset] = []

            self.per_variable_losses[dataset] = dict()

            for madis_var in self.madis_vars_o:
                self.per_variable_losses[dataset][madis_var] = dict()
                for per_variable_metrics_type in self.per_variable_metrics_types:
                    self.per_variable_losses[dataset][madis_var][per_variable_metrics_type] = []

    def addLoss(self, loss, per_variable_loss, stationType):
        self.losses[stationType].append(loss)
        for madis_var in self.madis_vars_o:
            for per_variable_metrics_type in self.per_variable_metrics_types:
                self.per_variable_losses[stationType][madis_var][per_variable_metrics_type] \
                    .append(per_variable_loss[madis_var][per_variable_metrics_type])

    def report(self, epoch, lr):
        print('Epoch: %d train_loss[%.5f] valid_loss[%.5f]' % (epoch + 1, self.losses['train'][-1],
                                                               self.losses['val'][-1]), flush=True)
        print(' ', flush=True)

    def finish_run(self, best_metrics, figures_path):

        # ##### Save #####
        train_losses = np.array(self.losses['train'])
        valid_losses = np.array(self.losses['val'])
        ##### Plotting #####
        self.plot_metric(train_losses, valid_losses, 'MSE', figures_path)

        for madis_var in self.madis_vars_o:
            for per_variable_metrics_type in self.per_variable_metrics_types:
                self.plot_metric(self.per_variable_losses['train'][madis_var][per_variable_metrics_type],
                                 self.per_variable_losses['val'][madis_var][per_variable_metrics_type],
                                 madis_var.name + ' ' + per_variable_metrics_type.name, figures_path)

    def plot_metric(self, train_losses, valid_losses, metric_name, fig_path, y_range=None):
        fig, axs = plt.subplots(1, 1, figsize=(10, 5))

        epochs = len(train_losses)
        axs.plot(np.arange(1, epochs + 1), train_losses, label='Train')
        axs.plot(np.arange(1, epochs + 1), valid_losses, label='Valid')

        if y_range != None:
            axs.set_ylim(y_range[0], y_range[1])

        axs.legend()
        axs.grid()
        axs.set_xlabel('Epochs')
        axs.set_ylabel(metric_name)

        axs.set_title(metric_name + ' Plot')

        plt.savefig(fig_path / ('_'.join(metric_name.split(' ')) + '_plot.png'))
        plt.close()


class WBTelemetry(Telemetry):
    def __init__(self, madis_vars_o, per_variable_metrics_types, args, output_saving_path):
        super().__init__(madis_vars_o, per_variable_metrics_types)

        import wandb

        wandb_output_dir = output_saving_path.parent / 'wandb_log'
        wandb_output_dir.mkdir(parents=True, exist_ok=True)
        wandb_output_dir = wandb_output_dir.absolute()

        self.wb = wandb.init(
            project='',
            entity='',
            config=args,
            dir=wandb_output_dir,
        )

    def report(self, epoch, lr):
        super().report(epoch, lr)

        message = {
            "loss/train": self.losses['train'][-1],
            "loss/val": self.losses['val'][-1]
        }

        for madis_var in self.madis_vars_o:
            for per_variable_metrics_type in self.per_variable_metrics_types:
                for stationType in ['train', 'val']:
                    val = self.per_variable_losses[stationType][madis_var][per_variable_metrics_type][-1]
                    message[f"{madis_var.name}_{per_variable_metrics_type.name}/{stationType}"] = val

        self.wb.log(message)

    def finish_run(self, best_metrics, figures_path):
        super().finish_run(best_metrics, figures_path)

        for save_metric_type in best_metrics.keys():
            self.wb.summary[save_metric_type.name + '_loss'] = best_metrics[save_metric_type]['loss']

            per_variable_losses = best_metrics[save_metric_type]['per_variable_loss']
            for madis_var in self.madis_vars_o:
                variable_losses = per_variable_losses[madis_var]
                for per_variable_metrics_type in variable_losses.keys():
                    val = variable_losses[per_variable_metrics_type]
                    self.wb.summary[f"{madis_var.name}_{per_variable_metrics_type.name}"] = val

        self.wb.finish()
