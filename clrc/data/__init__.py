from clrc.data.cub import (
    CUBDataModule,
    CUBDataModuleUnlabeled, CUBDataModuleCluster, CUBDataModuleCCLK, CUBDataModuleAttribute
)
from clrc.data.fitzpatrick import (
    FitzpatrickDataModule,
    FitzpatrickDataModuleLabeled, FitzpatrickDataModuleUnlabeled, FitzpatrickDataModuleAttribute,
)
from clrc.data.ilsvrc100 import (
    ILSVRC100DataModule,
    ILSVRC100DataModuleUnlabeled, ILSVRC100DataModuleCluster, ILSVRC100DataModuleAttribute
)
from clrc.data.utzap import (
    UTZapposDataModule,
    UTZapposDataModuleUnlabeled, UTZapposDataModuleCluster, UTZapposDataModuleCCLK, UTZapposDataModuleAttribute
)
from clrc.data.wider import (
    WIDERDataModule,
    WIDERDataModuleUnlabeled, WIDERDataModuleCluster, WIDERDataModuleCCLK, WIDERDataModuleAttribute
)

__all__ = [
    "CUBDataModule",
    "CUBDataModuleUnlabeled", "CUBDataModuleCluster", "CUBDataModuleCCLK", "CUBDataModuleAttribute",

    "FitzpatrickDataModule",
    "FitzpatrickDataModuleLabeled", "FitzpatrickDataModuleUnlabeled", "FitzpatrickDataModuleAttribute",

    "ILSVRC100DataModule",
    "ILSVRC100DataModuleUnlabeled", "ILSVRC100DataModuleCluster", "ILSVRC100DataModuleAttribute",

    "UTZapposDataModule",
    "UTZapposDataModuleUnlabeled", "UTZapposDataModuleCluster", "UTZapposDataModuleCCLK", "UTZapposDataModuleAttribute",

    "WIDERDataModule",
    "WIDERDataModuleUnlabeled", "WIDERDataModuleCluster", "WIDERDataModuleCCLK", "WIDERDataModuleAttribute",

]
