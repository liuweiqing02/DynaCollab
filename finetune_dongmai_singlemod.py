# coding=utf-8
"""
dongmai 单模态独立微调脚本（方案A：运行两次分别训练两个模态）
- 第一次：只训练 mod1（例如 CT）
- 第二次：只训练 mod2（例如 MR）

复用你现有工程：
- config.py: Config, FINE_TUNING
- dataset.py: Dataset_single, split_dataset, collate_fn
- losses.py: CombinedLoss
- metrics.py: SegmentationMetrics（或你自己的指标实现）
- models: UNet / VTUNet / UNETR / UniUnet / CrossModalUNet(不建议单模态) 等（按 config.model）

Batch 约定（你现在 dongmai 的 dataset/CompatibleModel 里就是这个）：
(mod1, mod2, mod1_label, mod2_label, id)

运行示例：
1) 只训练 CT（mod1）：
   python finetune_dongmai_singlemod.py --modality mod1 --tag CT
2) 只训练 MR（mod2）：
   python finetune_dongmai_singlemod.py --modality mod2 --tag MR
"""

import argparse
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader, RandomSampler, Subset
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, ChainedScheduler
from tqdm import tqdm

from config import Config, FINE_TUNING
from dataset import Dataset_single, split_dataset
from losses import CombinedLoss
from metrics import SegmentationMetrics, BraTSMetrics  # BraTSMetrics 不一定用到，保留也无妨


def init_logger(logger, log_file_path: str):
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        file_handler = logging.FileHandler(log_file_path, mode="w")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(message)s"))

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)


def build_model_from_config(config):
    """
    单模态训练：模型输入通道固定为 1（因为只输入一个模态）。
    因此这里会根据 config.model 构建“单输入单输出seg模型”。
    """
    model_name = str(config.model).lower()

    if model_name == "unet":
        from models.unet import UNet
        return UNet(config.num_classes, mode="seg")

    if model_name == "vtunet":
        from models.vision_transformer import VTUNet
        net = VTUNet(num_classes=config.num_classes, embed_dim=96, in_chans=1)
        try:
            net.load_from()
        except Exception as e:
            print(f"[WARN] VTUNet.load_from() failed, continue without it. err={e}")
        return net

    if model_name == "unetr":
        from models.UNETR import UNETR
        # 单模态 => input_dim=1
        return UNETR(img_shape=(128, 128, 128), output_dim=config.num_classes, input_dim=1)

    if model_name == "uniunet":
        from models.uniunet import UniUnet
        # 你原 main.py: UniUnet(in_channels=1,...)
        return UniUnet(in_channels=1, out_channels=config.num_classes, input_format="bcdhw")

    # CrossModalUNet 本身是双模态设计（mode="seg"时可能也支持num_modalities=2），
    # 但“单模态独立训练”不建议用它；如果你确实要用，需你确认其 forward 接口
    if model_name == "crossmodalunet":
        raise ValueError("CrossModalUNet is designed for multi-modality fusion; not recommended for single-modality run.")

    raise ValueError(f"Unknown model in config.model: {config.model}")


def _squeeze_label_to_bdhw(y: torch.Tensor) -> torch.Tensor:
    # [B,1,D,H,W] -> [B,D,H,W]
    if y.ndim == 5 and y.size(1) == 1:
        y = y.squeeze(1)
    return y.long()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", type=str, required=True, choices=["mod1", "mod2"],
                        help="Train only one modality: mod1 or mod2")
    parser.add_argument("--tag", type=str, default="",
                        help="A tag appended to checkpoint/log names, e.g. CT or MR")
    args = parser.parse_args()

    config = Config(FINE_TUNING)

    if config.data != "dongmai":
        raise ValueError(f"This script is dongmai-only, but config.data={config.data}")

    # 单模态训练 => 输入通道必须为 1
    config.in_channels = 1

    tag = args.tag.strip()
    if tag == "":
        tag = args.modality  # 默认用 mod1/mod2

    # Give each modality/tag an isolated run folder
    base_run_dir = config.run_dir
    config.run_dir = os.path.join(base_run_dir, f"singlemod_{tag}")
    config.log_dir = os.path.join(config.run_dir, "logs")
    config.checkpoint_dir = os.path.join(config.run_dir, "checkpoints")
    config.latest_checkpoint_path = os.path.join(config.checkpoint_dir, "latest.pth")
    config.best_checkpoint_path = os.path.join(config.checkpoint_dir, "best.pth")
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    log_file = os.path.join(config.log_dir, "training.log")

    logger = logging.getLogger(f"DM-SingleModFT-{tag}")
    init_logger(logger, log_file)

    device = torch.device("cuda" if (config.cuda and torch.cuda.is_available()) else "cpu")
    if config.cuda and not torch.cuda.is_available():
        raise ValueError("No GPU found: set cuda=False in config.py")

    logger.info(f"Device: {device}")
    logger.info(f"Dataset: {config.data} | Training only {args.modality} (tag={tag})")
    logger.info(f"Model: {config.model} | in_channels=1 | num_classes={config.num_classes}")

    # Dataset / Split
    dataset_train_full = Dataset_single(config, training=True)
    dataset_val_full = Dataset_single(config, training=False)
    if len(dataset_train_full) != len(dataset_val_full):
        raise ValueError(
            f"Train/val dataset size mismatch: "
            f"train={len(dataset_train_full)} val={len(dataset_val_full)}"
        )
    train_ratio = float(getattr(config, "train_ratio", 0.8))
    split_seed = int(getattr(config, "split_seed", 42))
    dataset_train, dataset_val_from_train = split_dataset(dataset_train_full, train_ratio=train_ratio, seed=split_seed)
    dataset_val = Subset(dataset_val_full, dataset_val_from_train.indices)

    worker_count = int(config.num_cpu_workers)
    loader_kwargs = dict(
        pin_memory=config.pin_mem,
        num_workers=worker_count,
    )
    if worker_count > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True

    loader_train = DataLoader(
        dataset_train,
        batch_size=config.batch_size,
        sampler=RandomSampler(dataset_train),
        collate_fn=dataset_train_full.collate_fn,
        **loader_kwargs,
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=config.batch_size_val,
        sampler=RandomSampler(dataset_val),
        collate_fn=dataset_train_full.collate_fn,
        **loader_kwargs,
    )

    # Model / Loss / Opt / Sched
    model = build_model_from_config(config).to(device)
    loss_fn = CombinedLoss(num_classes=config.num_classes, ce_weight=0.2, dice_weight=0.8)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=10)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, config.nb_epochs - 10))
    scheduler = ChainedScheduler([warmup_scheduler, cosine_scheduler])

    # Metrics：dongmai 用 SegmentationMetrics（与你 CompatibleModel 一致）
    skip_metrics_epochs = 60
    start_epoch = 1
    best_val_dice = 0.0
    best_val_loss = 1e9

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    latest_path = config.latest_checkpoint_path

    # 自动断点续训：如果同名 latest 存在则恢复（两次训练互不影响，因为 tag 不同）
    if os.path.exists(latest_path):
        logger.info(f"Auto-resume from existing latest: {latest_path}")
        ckpt = torch.load(latest_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_dice = float(ckpt.get("best_val_dice", best_val_dice))
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))

    for epoch in range(start_epoch, config.nb_epochs + 1):
        skip_metrics = epoch <= skip_metrics_epochs

        train_metrics_calc = SegmentationMetrics(config.num_classes, spacing=config.desired_spacing)
        val_metrics_calc = SegmentationMetrics(config.num_classes, spacing=config.desired_spacing)

        # ---------------- Train ----------------
        model.train()
        train_loss = 0.0
        pbar = tqdm(total=len(loader_train), desc=f"Train[{tag}] {epoch}/{config.nb_epochs}")
        for batch in loader_train:
            pbar.update()

            mod1, mod2, y1, y2, sample_id = batch

            x = mod1 if args.modality == "mod1" else mod2
            y = y1 if args.modality == "mod1" else y2  # dongmai 两个模态各自 label（你数据集是这样）

            x = x.to(device)  # [B,1,D,H,W]
            y = _squeeze_label_to_bdhw(y.to(device))  # [B,D,H,W]

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)  # [B,C,D,H,W]
            loss = loss_fn(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += float(loss) / max(1, len(loader_train))

            if not skip_metrics:
                with torch.no_grad():
                    dice, iou, hd95 = train_metrics_calc(logits, y)
                    train_metrics_calc.update(dice, iou, hd95)

        pbar.close()

        # ---------------- Val ----------------
        model.eval()
        val_loss = 0.0
        pbar = tqdm(total=len(loader_val), desc=f"Val[{tag}] {epoch}/{config.nb_epochs}")
        with torch.no_grad():
            for batch in loader_val:
                pbar.update()

                mod1, mod2, y1, y2, sample_id = batch
                x = mod1 if args.modality == "mod1" else mod2
                y = y1 if args.modality == "mod1" else y2

                x = x.to(device)
                y = _squeeze_label_to_bdhw(y.to(device))

                logits = model(x)
                loss = loss_fn(logits, y)
                val_loss += float(loss) / max(1, len(loader_val))

                if not skip_metrics:
                    dice, iou, hd95 = val_metrics_calc(logits, y)
                    val_metrics_calc.update(dice, iou, hd95)

        pbar.close()

        if scheduler is not None:
            scheduler.step()

        # ---------------- Log ----------------
        if skip_metrics:
            logger.info(
                f"[{tag}] Epoch [{epoch}/{config.nb_epochs}] (Skipping metrics)\n"
                f"Train Loss: {train_loss:.4f}\n"
                f"Val Loss: {val_loss:.4f}"
            )
        else:
            train_stats = train_metrics_calc.compute_stats()
            val_stats = val_metrics_calc.compute_stats()

            msg = (
                f"[{tag}] Epoch [{epoch}/{config.nb_epochs}]\n"
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n"
                f"Train Dice: {train_stats['dice']['str']} | Val Dice: {val_stats['dice']['str']}\n"
                f"Train IoU: {train_stats['iou']['str']} | Val IoU: {val_stats['iou']['str']}\n"
                f"Train HD95: {train_stats['hd95']['str']} | Val HD95: {val_stats['hd95']['str']}\n"
            )
            logger.info(msg)

        # ---------------- Save ----------------
        if not skip_metrics:
            current_val_dice = float(val_metrics_calc.compute_stats()["dice"]["mean"])
            if current_val_dice > best_val_dice:
                best_val_dice = current_val_dice
                best_path = config.best_checkpoint_path
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "best_val_dice": best_val_dice,
                        "best_val_loss": best_val_loss,
                        "tag": tag,
                        "modality": args.modality,
                        "model_name": config.model,
                    },
                    best_path,
                )
                logger.info(f"[{tag}] New BEST Val Dice={best_val_dice:.4f} -> saved: {best_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "best_val_dice": best_val_dice,
                "best_val_loss": best_val_loss,
                "tag": tag,
                "modality": args.modality,
                "model_name": config.model,
            },
            latest_path,
        )

    logger.info(f"[{tag}] Finished. Best Val Dice={best_val_dice:.4f} | Best Val Loss={best_val_loss:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)
