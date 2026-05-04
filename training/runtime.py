import logging
import os

from config import PRETRAINING
from trainer import Trainer
from training.builders import build_dataloaders, build_loss, build_model, get_dataset_class


def init_logger(logger, config):
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return

    config.ensure_output_dirs()
    file_handler = logging.FileHandler(os.path.join(config.log_dir, "public_preview.log"), mode="w")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def dump_run_metadata(config, logger):
    config_path = os.path.join(config.run_dir, "config_snapshot.json")
    config.dump_json(config_path)
    logger.info(f"Run dir: {config.run_dir}")
    logger.info(f"Config snapshot: {config_path}")


def run_training(config):
    logger = logging.getLogger("DynaCollab")
    init_logger(logger, config)
    dump_run_metadata(config, logger)

    dataset_cls = get_dataset_class(config)
    model = build_model(config)
    loss = build_loss(config)
    loaders = build_dataloaders(None, None, None, config)
    trainer = Trainer(model, loss, loaders["train_loader"], loaders["val_loader"], config)

    logger.info("DynaCollab public preview")
    logger.info(config.release_note)
    logger.info(f"Dataset interface: {dataset_cls.__name__}")
    logger.info(f"Model interface: {model.__class__.__name__}")
    logger.info(f"Loss interface: {loss.__class__.__name__}")
    logger.info("Training outline:")

    steps = trainer.pretraining() if config.mode == PRETRAINING else trainer.fine_tuning()
    for index, step in enumerate(steps, start=1):
        logger.info(f"{index}. {step}")

    logger.info("This repository version is documentation-oriented and not intended for full experiment reproduction.")
