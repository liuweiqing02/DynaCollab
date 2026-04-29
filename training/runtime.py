import logging
import os

from torch.utils.data import Subset

from config import PRETRAINING
from dataset import split_dataset
from trainer import Trainer
from training.builders import build_dataloaders, build_loss, build_model_pair, get_dataset_class


def init_logger(logger, config):
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return

    config.ensure_output_dirs()
    file_handler = logging.FileHandler(os.path.join(config.log_dir, "training.log"), mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def dump_run_metadata(config, logger):
    config_path = os.path.join(config.run_dir, "config_snapshot.json")
    config.dump_json(config_path)
    logger.info(f"Run dir: {config.run_dir}")
    logger.info(f"Config snapshot: {config_path}")


def save_validation_ids(dataset, dataset_val, config):
    val_ids = []
    for i in range(len(dataset_val)):
        try:
            original_idx = dataset_val.indices[i] if hasattr(dataset_val, "indices") else i
            val_ids.append(dataset.get_id(original_idx))
        except Exception:
            continue
    with open(os.path.join(config.log_dir, "validation_ids.csv"), "w", encoding="utf-8") as f:
        f.write("id\n")
        for val_id in val_ids:
            f.write(f"{val_id}\n")


def build_train_val_datasets(dataset_cls, config):
    train_dataset_full = dataset_cls(config, training=True)
    val_dataset_full = dataset_cls(config, training=False)
    if len(train_dataset_full) != len(val_dataset_full):
        raise ValueError(
            f"Train/val dataset size mismatch: "
            f"train={len(train_dataset_full)} val={len(val_dataset_full)}"
        )

    train_ratio = float(getattr(config, "train_ratio", 0.8))
    split_seed = int(getattr(config, "split_seed", 42))
    train_subset, val_subset_from_train = split_dataset(
        train_dataset_full, train_ratio=train_ratio, seed=split_seed
    )
    val_subset = Subset(val_dataset_full, val_subset_from_train.indices)
    return train_dataset_full, val_dataset_full, train_subset, val_subset


def run_training(config):
    logger = logging.getLogger("DynaCollab")
    init_logger(logger, config)
    dump_run_metadata(config, logger)

    dataset_cls = get_dataset_class(config)
    try:
        dataset_train_full, dataset_val_full, dataset_train, dataset_val = build_train_val_datasets(dataset_cls, config)
        logger.info(f"Loaded dataset {config.data}, total={len(dataset_train_full)}")
    except Exception as e:
        logger.error(f"Dataset init failed: {e}")
        raise

    logger.info(f"Split -> train={len(dataset_train)}, val={len(dataset_val)}")
    save_validation_ids(dataset_val_full, dataset_val, config)

    loader_train, loader_val = build_dataloaders(dataset_train_full, dataset_train, dataset_val, config)
    net_1, net_2 = build_model_pair(config)
    loss = build_loss(config)
    model = Trainer(net_1, net_2, loss, loader_train, loader_val, config, dataset_val=dataset_val)

    if config.mode == PRETRAINING:
        model.pretraining()
    else:
        model.fine_tuning()
