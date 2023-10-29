import pytorch_lightning as pl
import torch
import torch.optim as optim

from clrc.lars import LARS
from clrc.lr_scheduler import WarmupCosineLR, WarmupStepLR
from clrc.utils import calculate_train_steps


def _get_parameter_names(model, forbidden_layer_types, ignored_modules=None):
    """ Returns the names of the model parameters that are not inside a forbidden layer.
    Adapted from https://github.com/huggingface/transformers/blob/87d5057d863c927e31761acd00a6716653275931/src/
    transformers/trainer_pt_utils.py#L976
    """
    result = []
    for name, child in model.named_children():
        if ignored_modules and name in ignored_modules:
            continue
        result += [
            f"{name}.{n}"
            for n in _get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())

    return result


class BaseModule(pl.LightningModule):
    args_schema = {
        "learning_rate": (float, 0.01, "Base learning rate."),
        "optimizer": (str, "sgd", "Choose from ['sgd', 'adam', 'adamw', 'lars']."),
        "weight_decay": (float, 5e-4, "Weight decay coefficient."),
        "momentum": (float, 0.9, "Momentum for SGD."),
        "nesterov": (bool, False, "Whether to use Nesterov SGD."),
        "warmup_epochs": (int, 0, "Number of epochs to linearly warmup to."),
        "lr_scheduler": (str, None, "Choose from [None, 'cosine', 'step']."),
        "step_decay_milestones": (str, None, "List of epoch milestones (separated by ',') to apply "
                                             "learning rate decay in step scheduler."),
        "step_decay_factor": (float, 0.1, "Multiplicative factor for learning rate decay in step scheduler.")
    }

    def __init__(self, learning_rate, optimizer, weight_decay, momentum, nesterov, warmup_epochs, lr_scheduler,
                 step_decay_milestones, step_decay_factor):
        super().__init__()
        hparams = {
            "learning_rate": learning_rate,
            "optimizer": optimizer,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "lr_scheduler": lr_scheduler
        }
        if optimizer == 'sgd':
            hparams.update({
                "momentum": momentum,
                "nesterov": nesterov,
            })
        elif optimizer == 'lars':
            hparams.update({
                "momentum": momentum
            })
        if lr_scheduler == 'step':
            hparams.update({
                "step_decay_milestones": step_decay_milestones,
                "step_decay_factor": step_decay_factor
            })
        self.save_hyperparameters(hparams)

    def _configure_parameter_groups(self):
        # Disable weight decay for normalization parameters and biases
        decay_parameters = _get_parameter_names(
            self,
            forbidden_layer_types=[torch.nn.GroupNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d]
        )
        decay_parameters = [name for name in decay_parameters if "bias" not in name and len(name.split('.')) > 1]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if n in decay_parameters],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]

        return optimizer_grouped_parameters

    def _get_optimizer(self):
        optimizer_grouped_parameters = self._configure_parameter_groups()

        # Initialize optimizer
        if self.hparams.optimizer == 'sgd':
            optimizer = optim.SGD(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                momentum=self.hparams.momentum,
                nesterov=self.hparams.nesterov
            )
        elif self.hparams.optimizer == 'adam':
            optimizer = optim.Adam(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate
            )
        elif self.hparams.optimizer == 'adamw':
            optimizer = optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate
            )
        elif self.hparams.optimizer == 'lars':
            optimizer = LARS(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                eta=1e-3,
                momentum=self.hparams.momentum
            )
        else:
            raise ValueError(f"Unknown optimizer: '{self.hparams.optimizer}'.")

        return optimizer

    def _get_lr_scheduler(self, optimizer):
        hparams = self.hparams
        steps_per_epoch, total_steps = calculate_train_steps(self)
        warmup_steps = hparams.warmup_epochs * steps_per_epoch

        # Initialize learning rate scheduler
        if hparams.lr_scheduler == 'cosine':
            lr_scheduler = WarmupCosineLR(optimizer, warmup_steps, total_steps)
        elif hparams.lr_scheduler == 'step':
            epoch_milestones = [int(e) for e in hparams.step_decay_milestones.split(',')]
            step_milestones = [e * steps_per_epoch for e in epoch_milestones]
            factor = hparams.step_decay_factor
            lr_scheduler = WarmupStepLR(optimizer, warmup_steps, step_milestones, factor)
        elif hparams.lr_scheduler is None:
            lr_scheduler = WarmupStepLR(optimizer, warmup_steps, [], 1)
        else:
            raise ValueError(f"Unknown learning rate scheduler: '{hparams.lr_scheduler}'.")

        return lr_scheduler

    def configure_optimizers(self):
        optimizer = self._get_optimizer()
        lr_scheduler = self._get_lr_scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

    def register_log_metrics(self, metrics):
        self.logger.log_hyperparams({**self.hparams, **self.trainer.datamodule.hparams}, metrics)
