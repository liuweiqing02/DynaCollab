# coding=utf-8
"""
BraTS19 独立微调脚本（单文件）
- 双模态输入：t1ce + flair（通道拼接）
- 单模型输出：一个 segmentation logits
- loss：复用已有 CombinedLoss
- metrics：复用已有 BraTSMetrics（Dice/IoU/HD95 + 区域指标）
- config：复用原来的 Config（请在 config.py 内设置 data='BraTs19'，model='VTUNet' or 'UNETR'）

Batch 约定（你已确认）：
(t1ce, flair, y1, y2, id) 其中 y1/y2 理论相同，训练默认用 y1。

运行：
python finetune_brats_fused.py
"""

import logging
import os
import sys

import torch
from torch.utils.data import DataLoader, RandomSampler, Subset
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, ChainedScheduler
from tqdm import tqdm

from config import Config, FINE_TUNING
from dataset import Dataset_BraTs19, split_dataset
from losses import CombinedLoss
from metrics import BraTSMetrics


def init_logger(logger, config):
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        os.makedirs(config.log_dir, exist_ok=True)

        file_handler = logging.FileHandler(os.path.join(config.log_dir, "training.log"), mode="w")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(message)s"))

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)


def build_model_from_config(config):
    model_name = str(config.model).lower()

    if model_name == "vtunet":
        from models.vision_transformer import VTUNet
        net = VTUNet(num_classes=config.num_classes, embed_dim=96, in_chans=config.in_channels)
        try:
            net.load_from()
        except Exception as e:
            print(f"[WARN] VTUNet.load_from() failed, continue without it. err={e}")
        return net

    if model_name == "unetr":
        from models.UNETR import UNETR
        net = UNETR(img_shape=(128, 128, 128), output_dim=config.num_classes, input_dim=config.in_channels)
        return net

    if model_name == "uniunet":
        from models.uniunet import UniUnet
        net = UniUnet(
            in_channels=config.in_channels,
            out_channels=config.num_classes,
            input_format="bcdhw"
        )
        return net

    raise ValueError(f"Unknown model in config.model: {config.model}")


def _squeeze_label_to_bdhw(y: torch.Tensor) -> torch.Tensor:
    # [B,1,D,H,W] -> [B,D,H,W]
    if y.ndim == 5 and y.size(1) == 1:
        y = y.squeeze(1)
    return y.long()


def main():
    config = Config(FINE_TUNING)

    if config.data != "BraTs19":
        raise ValueError(f"This script is BraTs19-only, but config.data={config.data}")

    # 拼接输入 => 2通道
    if config.in_channels != 2:
        raise ValueError(
            f"For fused input (t1ce+flair), config.in_channels must be 2, but got {config.in_channels}. "
            f"Please set it in config.py for BraTs19."
        )

    logger = logging.getLogger("BraTS-FusedFT")
    init_logger(logger, config)

    device = torch.device("cuda" if (config.cuda and torch.cuda.is_available()) else "cpu")
    if config.cuda and not torch.cuda.is_available():
        raise ValueError("No GPU found: set cuda=False in config.py")

    logger.info(f"Device: {device}")
    logger.info(f"Dataset: {config.data} | Modalities: t1ce + flair (fused)")
    logger.info(f"Model: {config.model} | in_channels={config.in_channels} | num_classes={config.num_classes}")

    # Dataset / Split
    dataset_train_full = Dataset_BraTs19(config, training=True)
    dataset_val_full = Dataset_BraTs19(config, training=False)
    if len(dataset_train_full) != len(dataset_val_full):
        raise ValueError(
            f"Train/val dataset size mismatch: "
            f"train={len(dataset_train_full)} val={len(dataset_val_full)}"
        )
    train_ratio = float(getattr(config, "train_ratio", 0.8))
    split_seed = int(getattr(config, "split_seed", 42))
    dataset_train, dataset_val_from_train = split_dataset(dataset_train_full, train_ratio=train_ratio, seed=split_seed)
    dataset_val = Subset(dataset_val_full, dataset_val_from_train.indices)

    logger.info(
        f"Loaded dataset: total={len(dataset_train_full)}, train={len(dataset_train)}, val={len(dataset_val)}"
    )

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

    # Metrics（复用你 CompatibleModel 的逻辑）
    skip_metrics_epochs = 120

    start_epoch = 1
    best_val_dice = 0.0
    best_val_loss = 1e9

    # Resume: 断点续训（加载 optimizer/scheduler）
    if hasattr(config, "finetuning_checkpoint_path") and config.finetuning_checkpoint_path is not None:
        ckpt_path = config.finetuning_checkpoint_path
        logger.info(f"Resuming from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt and ckpt["optimizer"] is not None:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt and ckpt["scheduler"] is not None:
                scheduler.load_state_dict(ckpt["scheduler"])
            if "epoch" in ckpt:
                start_epoch = int(ckpt["epoch"]) + 1
            if "best_val_loss" in ckpt:
                best_val_loss = float(ckpt["best_val_loss"])
            if "best_val_dice" in ckpt:
                best_val_dice = float(ckpt["best_val_dice"])

        elif isinstance(ckpt, dict) and "model1" in ckpt:
            model.load_state_dict(ckpt["model1"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            if "epoch" in ckpt:
                start_epoch = int(ckpt["epoch"]) + 1
        else:
            raise ValueError("Unknown checkpoint format")

    # Pretrained: 只加载模型权重
    elif hasattr(config, "pretrained_path") and config.pretrained_path is not None:
        ckpt_path = config.pretrained_path
        logger.info(f"Loading pretrained weights from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict):
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"], strict=False)
            elif "model1" in ckpt:
                model.load_state_dict(ckpt["model1"], strict=False)
            else:
                model.load_state_dict(ckpt, strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, config.nb_epochs + 1):
        skip_metrics = epoch <= skip_metrics_epochs

        # 每个 epoch 重新建 metrics 计算器（与你 CompatibleModel 的风格一致）
        train_metrics_calc = BraTSMetrics(spacing=config.desired_spacing)
        val_metrics_calc = BraTSMetrics(spacing=config.desired_spacing)

        # -------------------------
        # Train
        # -------------------------
        model.train()
        nb_batch = len(loader_train)
        train_loss = 0.0

        pbar = tqdm(total=nb_batch, desc=f"Training {epoch}/{config.nb_epochs}")
        for batch in loader_train:
            pbar.update()

            t1ce, flair, y1, y2, sample_id = batch
            t1ce = t1ce.to(device)
            flair = flair.to(device)
            y1 = _squeeze_label_to_bdhw(y1.to(device))
            y2 = _squeeze_label_to_bdhw(y2.to(device))

            if not torch.equal(y1, y2):
                logger.warning("Found y1 != y2 for a batch. Using y1 as GT.")

            x = torch.cat([t1ce, flair], dim=1)  # [B,2,D,H,W]

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += float(loss) / max(1, nb_batch)

            if not skip_metrics:
                with torch.no_grad():
                    dice, iou, hd95 = train_metrics_calc(logits, y1)
                    train_metrics_calc.update(dice, iou, hd95)

        pbar.close()

        # -------------------------
        # Val
        # -------------------------
        model.eval()
        nb_batch = len(loader_val)
        val_loss = 0.0

        pbar = tqdm(total=nb_batch, desc=f"Validation {epoch}/{config.nb_epochs}")
        with torch.no_grad():
            for batch in loader_val:
                pbar.update()

                t1ce, flair, y1, y2, sample_id = batch
                t1ce = t1ce.to(device)
                flair = flair.to(device)
                y1 = _squeeze_label_to_bdhw(y1.to(device))

                x = torch.cat([t1ce, flair], dim=1)
                logits = model(x)
                loss = loss_fn(logits, y1)
                val_loss += float(loss) / max(1, nb_batch)

                if not skip_metrics:
                    dice, iou, hd95 = val_metrics_calc(logits, y1)
                    val_metrics_calc.update(dice, iou, hd95)

        pbar.close()

        if scheduler is not None:
            scheduler.step()

        # -------------------------
        # Logging
        # -------------------------
        if skip_metrics:
            logger.info(
                f"Epoch [{epoch}/{config.nb_epochs}] (Skipping metrics)\n"
                f"Train Loss: {train_loss:.4f}\n"
                f"Val Loss: {val_loss:.4f}"
            )
        else:
            train_stats = train_metrics_calc.compute_stats()
            val_stats = val_metrics_calc.compute_stats()

            # 兼容你 CompatibleModel 的打印格式（含 regions）
            log_msg = (
                f"Epoch [{epoch}/{config.nb_epochs}]\n"
                f"Train Loss: {train_loss:.4f}\n"
                f"Val Loss: {val_loss:.4f}\n"
                f"Train Dice: {train_stats['dice']['str']}\n"
                f"Val Dice: {val_stats['dice']['str']}\n"
                f"Train IoU: {train_stats['iou']['str']}\n"
                f"Val IoU: {val_stats['iou']['str']}\n"
                f"Train HD95: {train_stats['hd95']['str']}\n"
                f"Val HD95: {val_stats['hd95']['str']}\n"
            )

            # BraTS 区域指标（ET/TC/WT）
            if "regions" in val_stats.get("dice", {}):
                log_msg += (
                    f"Val Dice - ET: {val_stats['dice']['regions']['ET']['str']}, "
                    f"TC: {val_stats['dice']['regions']['TC']['str']}, "
                    f"WT: {val_stats['dice']['regions']['WT']['str']}\n"
                )
            if "regions" in val_stats.get("hd95", {}):
                log_msg += (
                    f"Val HD95 - ET: {val_stats['hd95']['regions']['ET']['str']}, "
                    f"TC: {val_stats['hd95']['regions']['TC']['str']}, "
                    f"WT: {val_stats['hd95']['regions']['WT']['str']}\n"
                )

            logger.info(log_msg)

        # -------------------------
        # Save ckpt
        # -------------------------
        # latest
        ckpt_latest = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "best_val_loss": best_val_loss,
            "best_val_dice": best_val_dice,
            "modalities": "t1ce+flair",
            "in_channels": config.in_channels,
            "num_classes": config.num_classes,
            "model_name": config.model,
        }
        latest_path = config.latest_checkpoint_path
        torch.save(ckpt_latest, latest_path)

        # best：用 val_dice（与你 CompatibleModel 类似）
        if not skip_metrics:
            val_stats = val_metrics_calc.compute_stats()
            current_val_dice = float(val_stats["dice"]["mean"])  # 按你 metrics 实现，这里应存在
            is_best = current_val_dice > best_val_dice
            if is_best:
                best_val_dice = current_val_dice
                best_path = config.best_checkpoint_path
                ckpt_best = dict(ckpt_latest)
                ckpt_best["best_val_dice"] = best_val_dice
                torch.save(ckpt_best, best_path)
                logger.info(f"New BEST Val Dice={best_val_dice:.4f} -> saved: {best_path}")

        # 同时记录 best_val_loss（可选）
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    logger.info(f"Finished. Best Val Dice={best_val_dice:.4f} | Best Val Loss={best_val_loss:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)
