import argparse

from clrc.config import get_argument_parser, parse_and_process_arguments, get_trainer
from clrc.data import (
    CUBDataModule, FitzpatrickDataModule, ILSVRC100DataModule, UTZapposDataModule, WIDERDataModule,

)
from clrc.model import Classifier, ClassifierFairness

DATA_MODULES = {
    "utzap": UTZapposDataModule,
    "wider": WIDERDataModule,
    "cub": CUBDataModule,
    "ilsvrc100": ILSVRC100DataModule,
    "fitzpatrick": FitzpatrickDataModule
}
LN_MODULES = {
    "classifier": Classifier,
    "classifier_fairness": ClassifierFairness
}
COMPATIBLE_LISTS = {
    "utzap": {"classifier"},
    "wider": {"classifier"},
    "cub": {"classifier"},
    "ilsvrc100": {"classifier"},
    "fitzpatrick": {"classifier_fairness"},
}


def check_compatibility(dm_name, model_name):
    if model_name not in COMPATIBLE_LISTS[dm_name]:
        raise ValueError(f"'{model_name}' is not compatible with '{dm_name}'.")


def main(_args, _dm_cls, _model_cls):
    data_args, model_args, trainer_args, other_args = _args
    dm = _dm_cls(**data_args)
    model = _model_cls(**model_args)
    trainer = get_trainer(**trainer_args)

    if other_args["do_train"]:
        # Train
        trainer.fit(model, dm)

    if other_args["do_test"]:
        # Test
        if not other_args["do_train"]:
            model = _model_cls.load_from_checkpoint(trainer_args["resume_from_checkpoint"], **model_args)
        trainer.test(model, dm)


if __name__ == "__main__":
    entry_parser = argparse.ArgumentParser(add_help=False)
    entry_parser.add_argument('data_module', type=str, choices=list(DATA_MODULES.keys()))
    entry_parser.add_argument('model', type=str, choices=list(LN_MODULES.keys()))
    entry_args = entry_parser.parse_known_args()[0]

    check_compatibility(entry_args.data_module, entry_args.model)

    dm_cls = DATA_MODULES[entry_args.data_module]
    model_cls = LN_MODULES[entry_args.model]

    parser = get_argument_parser(dm_cls, model_cls, parents=[entry_parser])
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    args = parse_and_process_arguments(parser)

    main(args, dm_cls, model_cls)
