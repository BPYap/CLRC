import argparse

from clrc.config import get_argument_parser, parse_and_process_arguments, get_trainer
from clrc.data import (
    CUBDataModule,
    CUBDataModuleUnlabeled, CUBDataModuleCluster, CUBDataModuleCCLK, CUBDataModuleAttribute,

    FitzpatrickDataModuleLabeled,
    FitzpatrickDataModuleUnlabeled, FitzpatrickDataModuleAttribute,

    ILSVRC100DataModule,
    ILSVRC100DataModuleUnlabeled, ILSVRC100DataModuleCluster, ILSVRC100DataModuleAttribute,

    UTZapposDataModule,
    UTZapposDataModuleUnlabeled, UTZapposDataModuleCluster, UTZapposDataModuleCCLK, UTZapposDataModuleAttribute,

    WIDERDataModule,
    WIDERDataModuleUnlabeled, WIDERDataModuleCluster, WIDERDataModuleCCLK, WIDERDataModuleAttribute,

)
from clrc.model import (
    SimCLRLearner, BYOLLearner, SwAVLearner, SupConLearner, CELearner, CMCLearner, CCLKLearner,
    CLRCLearner, HMCELearner
)

DATA_MODULES = {
    "cub": CUBDataModuleUnlabeled,
    "cub-labeled": CUBDataModule,
    "cub-cluster": CUBDataModuleCluster,
    "cub-cclk": CUBDataModuleCCLK,
    "cub-attribute": CUBDataModuleAttribute,

    "utzap": UTZapposDataModuleUnlabeled,
    "utzap-labeled": UTZapposDataModule,
    "utzap-cluster": UTZapposDataModuleCluster,
    "utzap-cclk": UTZapposDataModuleCCLK,
    "utzap-attribute": UTZapposDataModuleAttribute,

    "wider": WIDERDataModuleUnlabeled,
    "wider-labeled": WIDERDataModule,
    "wider-cluster": WIDERDataModuleCluster,
    "wider-cclk": WIDERDataModuleCCLK,
    "wider-attribute": WIDERDataModuleAttribute,

    "ilsvrc100": ILSVRC100DataModuleUnlabeled,
    "ilsvrc100-labeled": ILSVRC100DataModule,
    "ilsvrc100-cluster": ILSVRC100DataModuleCluster,
    "ilsvrc100-attribute": ILSVRC100DataModuleAttribute,

    "fitzpatrick": FitzpatrickDataModuleUnlabeled,
    "fitzpatrick-labeled": FitzpatrickDataModuleLabeled,
    "fitzpatrick-attribute": FitzpatrickDataModuleAttribute,
}
LN_MODULES = {
    "simclr": SimCLRLearner,
    "byol": BYOLLearner,
    "swav": SwAVLearner,
    "supcon": SupConLearner,
    "ce": CELearner,
    "cmc": CMCLearner,
    "cclk": CCLKLearner,
    "clrc": CLRCLearner,
    "hmce": HMCELearner
}
COMPATIBLE_LISTS = {
    "cub": {"simclr", "byol", "swav"},
    "cub-labeled": {"supcon"},
    "cub-cluster": {"supcon"},
    "cub-cclk": {"cclk"},
    "cub-attribute": {"ce", "cmc", "clrc"},

    "utzap": {"simclr", "byol", "swav"},
    "utzap-labeled": {"supcon"},
    "utzap-cluster": {"supcon"},
    "utzap-cclk": {"cclk"},
    "utzap-attribute": {"ce", "cmc", "clrc"},

    "wider": {"simclr", "byol", "swav"},
    "wider-labeled": {"supcon"},
    "wider-cluster": {"supcon"},
    "wider-cclk": {"cclk"},
    "wider-attribute": {"ce", "cmc", "clrc"},

    "ilsvrc100": {"simclr", "byol", "swav"},
    "ilsvrc100-labeled": {"supcon"},
    "ilsvrc100-cluster": {"supcon"},
    "ilsvrc100-attribute": {"ce", "cmc", "cclk", "clrc", "hmce"},

    "fitzpatrick": {"simclr", "byol", "swav"},
    "fitzpatrick-labeled": {"supcon"},
    "fitzpatrick-attribute": {"ce", "cmc", "cclk", "clrc"}
}


def check_compatibility(dm_name, model_name):
    if model_name not in COMPATIBLE_LISTS[dm_name]:
        raise ValueError(f"'{model_name}' is not compatible with '{dm_name}'.")


def main(_args, _dm_cls, _model_cls):
    data_args, model_args, trainer_args, other_args = _args
    dm = _dm_cls(**data_args)
    model = _model_cls(**model_args)
    trainer = get_trainer(**trainer_args)

    trainer.fit(model, dm)


if __name__ == "__main__":
    entry_parser = argparse.ArgumentParser(add_help=False)
    entry_parser.add_argument('data_module', type=str, choices=list(DATA_MODULES.keys()))
    entry_parser.add_argument('model', type=str, choices=list(LN_MODULES.keys()))
    entry_args = entry_parser.parse_known_args()[0]

    check_compatibility(entry_args.data_module, entry_args.model)

    dm_cls = DATA_MODULES[entry_args.data_module]
    model_cls = LN_MODULES[entry_args.model]

    parser = get_argument_parser(dm_cls, model_cls, parents=[entry_parser])
    args = parse_and_process_arguments(parser)

    main(args, dm_cls, model_cls)
