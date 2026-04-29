from torch.utils.data import DataLoader, RandomSampler

from config import PRETRAINING
from losses import CombinedGlobalLocalLoss, CombinedLoss, DualModalContrastiveLoss
from models.dynacollab import CrossModalUNet


def get_dataset_class(config):
    if config.data == "BraTs19":
        from dataset import DatasetBraTS19

        return DatasetBraTS19
    if config.data == "carotid":
        from dataset import DatasetCarotid

        return DatasetCarotid
    raise ValueError(f"Unsupported dataset: {config.data}")


def build_dataloaders(dataset, dataset_train, dataset_val, config):
    worker_count = int(config.num_cpu_workers)
    common_kwargs = dict(
        collate_fn=dataset.collate_fn,
        pin_memory=config.pin_mem,
        num_workers=worker_count,
    )
    if worker_count > 0:
        common_kwargs["prefetch_factor"] = 2
        common_kwargs["persistent_workers"] = True

    loader_train = DataLoader(
        dataset_train,
        batch_size=config.batch_size,
        sampler=RandomSampler(dataset_train),
        **common_kwargs,
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=config.batch_size_val,
        sampler=RandomSampler(dataset_val),
        **common_kwargs,
    )
    return loader_train, loader_val


def build_model_pair(config):
    mode = "pretrain" if config.mode == PRETRAINING else "seg"
    return (
        CrossModalUNet(
            config.num_classes,
            in_channels=config.in_channels,
            mode=mode,
            num_modalities=2,
            growth_rate=config.growth_rate,
            use_anatomical_alignment=config.use_anatomical_alignment,
            use_global_local_loss=config.use_global_local_loss,
            baseline=config.baseline,
            use_cross_align=config.use_cross_align,
        ),
        None,
    )


def build_loss(config):
    if config.mode == PRETRAINING:
        if config.pretrain_loss == "tstcl":
            return CombinedGlobalLocalLoss(
                global_temp=0.1,
                local_temp=0.1,
                boundary_dilation=2,
                max_samples=20000,
                global_weight=0.4,
                local_weight=0.6,
            )
        if config.pretrain_loss == "contrastive":
            return DualModalContrastiveLoss()
        raise ValueError(f"Unsupported pretraining loss: {config.pretrain_loss}")
    return CombinedLoss(num_classes=config.num_classes, ce_weight=0.2, dice_weight=0.8)
