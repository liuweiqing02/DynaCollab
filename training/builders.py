from torch.utils.data import DataLoader, RandomSampler

from config import PRETRAINING
from losses import CombinedGlobalLocalLoss, CombinedLoss, DualModalContrastiveLoss
from models.CrossModalUNet import CrossModalUNet
from models.unet import UNet


def get_dataset_class(config):
    if config.data == "BraTs19":
        from dataset import Dataset_BraTs19

        return Dataset_BraTs19
    if config.data == "dongmai":
        from dataset import Dataset_single

        return Dataset_single
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
    if config.mode == PRETRAINING:
        return _build_pretraining_model_pair(config)
    return _build_finetuning_model_pair(config)


def _build_cross_modal_unet(config, mode):
    return CrossModalUNet(
        config.num_classes,
        in_channels=config.in_channels,
        mode=mode,
        num_modalities=2,
        growth_rate=config.growth_rate,
        use_anatomical_alignment=config.use_anatomical_alignment,
        use_global_local_loss=config.use_global_local_loss,
        baseline=config.baseline,
        use_cross_align=config.use_cross_align,
    )


def _build_pretraining_model_pair(config):
    model_name = str(config.model)
    if model_name == "CrossModalUNet":
        # Single-stream model that jointly consumes [mod1, mod2].
        return _build_cross_modal_unet(config, mode="pretrain"), None

    if model_name == "UNet":
        if config.use_global_local_loss:
            raise ValueError(
                "UNet pretraining does not support use_global_local_loss=True. "
                "Use model=CrossModalUNet or set use_global_local_loss=False."
            )
        # Dual-stream setup: one encoder per modality.
        return UNet(config.num_classes, mode="simCLR"), UNet(config.num_classes, mode="simCLR")

    raise ValueError(f"Unknown pretraining model: {model_name}")


def _build_finetuning_model_pair(config):
    model_name = str(config.model)

    if model_name == "CrossModalUNet":
        # Single-stream dual-input model.
        return _build_cross_modal_unet(config, mode="seg"), None

    if model_name == "UNet":
        return UNet(config.num_classes, mode="seg"), UNet(config.num_classes, mode="seg")

    if model_name == "VTUNet":
        from models.vision_transformer import VTUNet

        net_1 = VTUNet(num_classes=config.num_classes, embed_dim=84)
        net_1.load_from()
        net_2 = VTUNet(num_classes=config.num_classes, embed_dim=84)
        net_2.load_from()
        return net_1, net_2

    if model_name == "UNETR":
        from models.UNETR import UNETR

        return (
            UNETR(img_shape=(128, 128, 128), output_dim=config.num_classes, input_dim=1),
            UNETR(img_shape=(128, 128, 128), output_dim=config.num_classes, input_dim=1),
        )

    if model_name == "UniUnet":
        from models.uniunet import UniUnet

        return (
            UniUnet(in_channels=1, out_channels=config.num_classes, input_format="bcdhw"),
            UniUnet(in_channels=1, out_channels=config.num_classes, input_format="bcdhw"),
        )

    raise ValueError(f"Unknown finetuning model: {model_name}")


def build_loss(config):
    if config.mode == PRETRAINING:
        if config.use_global_local_loss:
            return CombinedGlobalLocalLoss(
                global_temp=0.1,
                local_temp=0.1,
                boundary_dilation=2,
                max_samples=20000,
                global_weight=0.4,
                local_weight=0.6,
            )
        return DualModalContrastiveLoss()
    return CombinedLoss(num_classes=config.num_classes, ce_weight=0.2, dice_weight=0.8)
