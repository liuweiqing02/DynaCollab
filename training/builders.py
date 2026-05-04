from config import PRETRAINING
from dataset import DatasetBraTS19, DatasetCarotid
from losses import CombinedGlobalLocalLoss, CombinedLoss, DualModalContrastiveLoss
from models.dynacollab import CrossModalUNet


def get_dataset_class(config):
    if config.data == "BraTs19":
        return DatasetBraTS19
    if config.data == "carotid":
        return DatasetCarotid
    raise ValueError(f"Unsupported dataset: {config.data}")


def build_dataloaders(dataset, dataset_train, dataset_val, config):
    return {
        "train_loader": "withheld",
        "val_loader": "withheld",
        "note": "Data loading and preprocessing are not included in the public preview.",
    }


def build_model(config):
    mode = "pretrain" if config.mode == PRETRAINING else "seg"
    return CrossModalUNet(
        num_classes=config.num_classes,
        in_channels=config.in_channels,
        mode=mode,
        num_modalities=config.num_modalities,
        baseline=config.baseline,
        use_cross_align=config.use_cross_align,
    )


def build_loss(config):
    if config.mode == PRETRAINING:
        if config.pretrain_loss == "tstcl":
            return CombinedGlobalLocalLoss()
        return DualModalContrastiveLoss()
    return CombinedLoss(num_classes=config.num_classes)
