import argparse
import json
import os
import sys
from collections import defaultdict

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback,  LearningRateMonitor, EarlyStopping
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities import rank_zero_only

from clrc.logger import get_logger


def _list_or_int(s):
    if s[0] == '[' and s[-1] == ']':
        return [int(c) for c in s[1:-1].split(',')]
    else:
        return int(s)


def _float_or_int(s):
    if len(s.split('.')) == 2:
        return float(s)
    else:
        return int(s)


TRAINER_ARGS = {
    # name:                    (type, default, help)
    "default_root_dir":        (str, os.getcwd(), "Default directory for logs and weights."),
    "resume_from_checkpoint":  (str, None, "Path to model checkpoint (.ckpt)."),
    "gpus":                    (_list_or_int, 0, "Specify number of GPUs to use."),
    "num_nodes":               (int, 1, "Number of GPU nodes for distributed training."),
    "accelerator":             (str, None, "Choose from [None, 'ddp']."),
    "amp_backend":             (str, "native", "Backend for mixed precision. Choose from ['native', 'apex']."),
    "amp_level":               (str, "O2", "Optimization level to use for apex amp."),
    "precision":               (int, 32, "Choose from [16, 32, 64]."),
    "accumulate_grad_batches": (int, 1, "Number of steps to accumulate gradients."),
    "max_epochs":              (int, 1000, "Number of epochs to train for."),
    "max_steps":               (int, None, "Number of steps to train for."),
    "check_val_every_n_epoch": (int, 1, "Run validation loop for every n-th epochs."),
    "val_check_interval":      (_float_or_int, 1.0, "Run validation loop for every n steps/fractions of train loop."),
    "log_every_n_steps":       (int, 50, "Number of logging steps."),
    "num_sanity_val_steps":    (int, 2, "Run n batches of validation samples as sanity check before starting the "
                                        "training routine."),
    "early_stopping_monitor":  (str, None, "Quantity to be monitored for early stopping"),
    "early_stopping_patience": (int, None, "Number of checks with no improvement before stopping the training.")
}

PARSED_ARGS = None


class _ArgumentParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.group_args = defaultdict(set)

    def add_args(self, group_name, args_schema):
        group = self.add_argument_group(group_name)
        for name, param in args_schema.items():
            type_, default, help_ = param
            if type_ == bool:
                group.add_argument(
                    f"--{name}",
                    action='store_true',
                    help=help_
                )
            else:
                group.add_argument(
                    f"--{name}",
                    type=type_,
                    default=default,
                    help=help_,
                    metavar=default
                )
            self.group_args[group_name].add(name)

    def get_args(self, group_name):
        return self.group_args[group_name]


def get_argument_parser(dm_cls, model_cls, include_trainer_args=True, **kwargs):
    parser = _ArgumentParser(**kwargs)
    parser.add_args("data_args", dm_cls.args_schema)
    parser.add_args("model_args", model_cls.args_schema)
    if include_trainer_args:
        parser.add_args("trainer_args", TRAINER_ARGS)

    # other args
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for reproducibility.",
        metavar=42
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        help="Load arguments from a configuration file (.json). Any optional"
             "argument specified in the command line will override the values"
             "from this file.",
    )

    return parser


def group_arguments(parser, parsed_arguments):
    data_args = dict()
    model_args = dict()
    trainer_args = dict()
    other_args = dict()

    for k, v in parsed_arguments.items():
        if k in parser.get_args("data_args"):
            data_args[k] = v
        elif k in parser.get_args("model_args"):
            model_args[k] = v
        elif k in parser.get_args("trainer_args"):
            trainer_args[k] = v
        else:
            other_args[k] = v

    return data_args, model_args, trainer_args, other_args


def parse_and_process_arguments(parser):
    args = parser.parse_args()
    if args.config_file:
        with open(args.config_file, 'r') as f:
            json_arguments = json.load(f)
        for k, v in vars(args).items():
            # override arguments from file with the ones provided in the
            # command line
            if f"--{k}" in sys.argv:
                json_arguments[k] = v
        grouped_args = group_arguments(parser, json_arguments)
    else:
        grouped_args = group_arguments(parser, vars(args))
    seed_everything(args.seed, workers=True)

    data_args, model_args, trainer_args, other_args = grouped_args
    global PARSED_ARGS
    PARSED_ARGS = {**data_args, **model_args, **trainer_args, **other_args}
    if "config_file" in PARSED_ARGS:
        del PARSED_ARGS["config_file"]

    return data_args, model_args, trainer_args, other_args


def get_trainer(**kwargs):
    class SnapshotCallback(Callback):
        def on_train_start(self, _trainer, pl_module):
            @rank_zero_only
            def save_snapshots():
                log_dir = _trainer.logger.log_dir
                # save command line arguments
                with open(os.path.join(log_dir, "args.json"), 'w') as f:
                    json.dump(
                        dict(sorted(PARSED_ARGS.items(), key=lambda x: x[0])),
                        f,
                        indent=4
                    )
            save_snapshots()

    callbacks = [
        SnapshotCallback(),
        LearningRateMonitor(logging_interval='step', log_momentum=True),
    ]
    if "early_stopping_monitor" in kwargs:
        if kwargs["early_stopping_monitor"] is not None:
            callbacks.append(
                EarlyStopping(
                    monitor=kwargs["early_stopping_monitor"],
                    patience=kwargs["early_stopping_patience"],
                    mode='max',
                    verbose=True
                )
            )
        del kwargs["early_stopping_monitor"]
        del kwargs["early_stopping_patience"]

    _kwargs = {
        "callbacks": callbacks,
        "checkpoint_callback": True,
        "log_gpu_memory": None,  # logging GPU memory might slow down training
        "logger": get_logger(kwargs["default_root_dir"]),
        "plugins": DDPPlugin(find_unused_parameters=False) if kwargs["accelerator"] == 'ddp' else None
    }
    _kwargs.update(kwargs)
    trainer = Trainer(**_kwargs)

    return trainer
