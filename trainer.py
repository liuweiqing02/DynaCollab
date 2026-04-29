import logging
import os

import torch
from torch.nn import DataParallel
from torch.optim.lr_scheduler import ChainedScheduler, CosineAnnealingLR, LinearLR
from tqdm import tqdm

from metrics import BraTSMetrics, SegmentationMetrics


class Trainer:
    def __init__(self, model, loss, loader_train, loader_val, config, dataset_val=None):
        self.model = model
        self.loss = loss
        self.loader = loader_train
        self.loader_val = loader_val
        self.config = config
        self.dataset_val = dataset_val
        self.logger = logging.getLogger("DynaCollab")
        self.device = torch.device("cuda" if config.cuda else "cpu")

        if config.cuda and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but no GPU is available. Set config.cuda=False to run on CPU.")

        self.model = DataParallel(self.model).to(self.device)
        self.optimizer = None
        self.scheduler = None
        self.best_val_dice = 0.0
        self.best_val_loss = float("inf")

    def _latest_ckpt_path(self):
        return getattr(
            self.config,
            "latest_checkpoint_path",
            os.path.join(self.config.checkpoint_dir, f"{self.config.checkpoint_name}_latest.pth"),
        )

    def _best_ckpt_path(self):
        return getattr(
            self.config,
            "best_checkpoint_path",
            os.path.join(self.config.checkpoint_dir, f"{self.config.checkpoint_name}_best.pth"),
        )

    def _configure_optimizer(self):
        params = list(self.model.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=self.config.lr, weight_decay=self.config.weight_decay)

        warmup_iters = min(10, max(1, self.config.nb_epochs))
        warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_iters)
        cosine = CosineAnnealingLR(self.optimizer, T_max=max(1, self.config.nb_epochs - warmup_iters))
        self.scheduler = ChainedScheduler([warmup, cosine])
        return params

    def _early_stopping_cfg(self):
        patience = int(getattr(self.config, "early_stopping_patience", 0))
        return {
            "enabled": bool(getattr(self.config, "use_early_stopping", False)) and patience > 0,
            "patience": max(patience, 0),
            "min_delta": max(float(getattr(self.config, "early_stopping_min_delta", 0.0)), 0.0),
            "start_epoch": max(int(getattr(self.config, "early_stopping_start_epoch", 1)), 1),
        }

    def _checkpoint(self, epoch):
        return {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "loss": self.loss.state_dict() if hasattr(self.loss, "state_dict") else None,
            "config": self.config.to_dict(),
        }

    def _save_checkpoint(self, checkpoint, is_best):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        torch.save(checkpoint, self._latest_ckpt_path())
        if is_best:
            torch.save(checkpoint, self._best_ckpt_path())

    def _pretraining_forward(self, mod1, mod2, mod1_label, mod2_label):
        if self.config.use_global_local_loss:
            global_projs, local_projs, deform_params = self.model([mod1, mod2])
            return self.loss(
                global_projs[0],
                global_projs[1],
                local_projs[0],
                local_projs[1],
                mod1_label,
                mod2_label,
                deform_params_list=deform_params,
            )

        projections = self.model([mod1, mod2])
        if len(projections) != 2:
            raise ValueError("Expected two modality projections for contrastive pretraining.")
        return self.loss(projections[0], projections[1])

    def _segmentation_forward(self, mod1, mod2):
        outputs = self.model([mod1, mod2])
        if len(outputs) != 2:
            raise ValueError("Expected two modality segmentation outputs.")
        return outputs[0], outputs[1]

    def pretraining(self):
        model_params = self._configure_optimizer()
        start_epoch = 1
        if getattr(self.config, "pretrained_checkpoint_path", None):
            start_epoch = self.load_model(self.config.pretrained_checkpoint_path)

        early_cfg = self._early_stopping_cfg()
        bad_epochs = 0
        self.logger.info(f"Pretraining with {self.config.pretrain_loss} loss from epoch {start_epoch}")

        for epoch in range(start_epoch, self.config.nb_epochs + 1):
            self.model.train()
            train_loss = 0.0
            grad_accum_steps = max(1, int(getattr(self.config, "grad_accum_steps", 1)))
            self.optimizer.zero_grad(set_to_none=True)
            pbar = tqdm(self.loader, desc=f"Pretrain {epoch}/{self.config.nb_epochs}")

            for batch_idx, batch in enumerate(pbar):
                mod1, mod2, mod1_label, mod2_label, _ = batch
                mod1 = mod1.to(self.device)
                mod2 = mod2.to(self.device)
                mod1_label = mod1_label.to(self.device)
                mod2_label = mod2_label.to(self.device)

                batch_loss = self._pretraining_forward(mod1, mod2, mod1_label, mod2_label)
                raw_loss = batch_loss.detach()
                (batch_loss / grad_accum_steps).backward()

                should_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(self.loader))
                if should_step:
                    torch.nn.utils.clip_grad_norm_(model_params, max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                train_loss += float(raw_loss) / max(1, len(self.loader))
                pbar.set_postfix(loss=f"{float(raw_loss):.4f}")

            val_loss = self._validate_pretraining()
            if self.scheduler is not None:
                self.scheduler.step()

            is_best = val_loss < (self.best_val_loss - early_cfg["min_delta"])
            if is_best:
                self.best_val_loss = val_loss
                bad_epochs = 0
            elif early_cfg["enabled"] and epoch >= early_cfg["start_epoch"]:
                bad_epochs += 1

            self._save_checkpoint(self._checkpoint(epoch), is_best=is_best)
            self.logger.info(
                f"Epoch {epoch}/{self.config.nb_epochs} - train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, best_val_loss={self.best_val_loss:.4f}"
            )

            if early_cfg["enabled"] and bad_epochs >= early_cfg["patience"]:
                self.logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

    def _validate_pretraining(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(self.loader_val, desc="Validate pretraining"):
                mod1, mod2, mod1_label, mod2_label, _ = batch
                mod1 = mod1.to(self.device)
                mod2 = mod2.to(self.device)
                mod1_label = mod1_label.to(self.device)
                mod2_label = mod2_label.to(self.device)
                batch_loss = self._pretraining_forward(mod1, mod2, mod1_label, mod2_label)
                val_loss += float(batch_loss) / max(1, len(self.loader_val))
        return val_loss

    def fine_tuning(self):
        model_params = self._configure_optimizer()
        start_epoch = 1

        if getattr(self.config, "pretrained_path", None):
            self._load_weights(self.config.pretrained_path, strict=False, load_optimizer=False)
        elif getattr(self.config, "finetuning_checkpoint_path", None):
            start_epoch = self.load_model(self.config.finetuning_checkpoint_path)

        metrics_mod1, metrics_mod2 = self._build_metrics()
        early_cfg = self._early_stopping_cfg()
        bad_epochs = 0
        self.logger.info(f"Fine-tuning from epoch {start_epoch}")

        for epoch in range(start_epoch, self.config.nb_epochs + 1):
            metrics_mod1.reset()
            metrics_mod2.reset()
            train_metrics = self._train_segmentation_epoch(epoch, model_params, metrics_mod1, metrics_mod2)
            val_metrics, val_stats_mod1, val_stats_mod2 = self._validate_segmentation_epoch(metrics_mod1, metrics_mod2)

            if self.scheduler is not None:
                self.scheduler.step()

            current_val_dice = (val_stats_mod1["dice"]["mean"] + val_stats_mod2["dice"]["mean"]) / 2
            is_best = current_val_dice > (self.best_val_dice + early_cfg["min_delta"])
            if is_best:
                self.best_val_dice = current_val_dice
                bad_epochs = 0
            elif early_cfg["enabled"] and epoch >= early_cfg["start_epoch"]:
                bad_epochs += 1

            self._save_checkpoint(self._checkpoint(epoch), is_best=is_best)
            self._log_segmentation_epoch(epoch, train_metrics, val_metrics, val_stats_mod1, val_stats_mod2)

            if early_cfg["enabled"] and bad_epochs >= early_cfg["patience"]:
                self.logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

    def _build_metrics(self):
        if self.config.data == "BraTs19":
            return BraTSMetrics(spacing=self.config.desired_spacing), BraTSMetrics(spacing=self.config.desired_spacing)
        return (
            SegmentationMetrics(self.config.num_classes, spacing=self.config.desired_spacing),
            SegmentationMetrics(self.config.num_classes, spacing=self.config.desired_spacing),
        )

    def _train_segmentation_epoch(self, epoch, model_params, metrics_mod1, metrics_mod2):
        self.model.train()
        metrics = {"loss": 0.0, "loss_mod1": 0.0, "loss_mod2": 0.0}
        grad_accum_steps = max(1, int(getattr(self.config, "grad_accum_steps", 1)))
        self.optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(self.loader, desc=f"Fine-tune {epoch}/{self.config.nb_epochs}")

        for batch_idx, batch in enumerate(pbar):
            mod1, mod2, mod1_label, mod2_label, _ = batch
            mod1 = mod1.to(self.device)
            mod2 = mod2.to(self.device)
            mod1_label = mod1_label.squeeze(1).long().to(self.device)
            mod2_label = mod2_label.squeeze(1).long().to(self.device)

            out1, out2 = self._segmentation_forward(mod1, mod2)
            loss1 = self.loss(out1, mod1_label)
            loss2 = self.loss(out2, mod2_label)
            batch_loss = loss1 + loss2
            raw_loss = batch_loss.detach()
            (batch_loss / grad_accum_steps).backward()

            should_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(self.loader))
            if should_step:
                torch.nn.utils.clip_grad_norm_(model_params, max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                self._update_metrics(metrics_mod1, out1, mod1_label)
                self._update_metrics(metrics_mod2, out2, mod2_label)

            metrics["loss"] += float(raw_loss) / max(1, len(self.loader))
            metrics["loss_mod1"] += float(loss1.detach()) / max(1, len(self.loader))
            metrics["loss_mod2"] += float(loss2.detach()) / max(1, len(self.loader))
            pbar.set_postfix(loss=f"{float(raw_loss):.4f}")

        return metrics

    def _validate_segmentation_epoch(self, metrics_mod1, metrics_mod2):
        self.model.eval()
        metrics_mod1.reset()
        metrics_mod2.reset()
        val_metrics = {"loss": 0.0, "loss_mod1": 0.0, "loss_mod2": 0.0}

        with torch.no_grad():
            for batch in tqdm(self.loader_val, desc="Validate fine-tuning"):
                mod1, mod2, mod1_label, mod2_label, _ = batch
                mod1 = mod1.to(self.device)
                mod2 = mod2.to(self.device)
                mod1_label = mod1_label.squeeze(1).long().to(self.device)
                mod2_label = mod2_label.squeeze(1).long().to(self.device)

                out1, out2 = self._segmentation_forward(mod1, mod2)
                loss1 = self.loss(out1, mod1_label)
                loss2 = self.loss(out2, mod2_label)
                batch_loss = loss1 + loss2

                self._update_metrics(metrics_mod1, out1, mod1_label)
                self._update_metrics(metrics_mod2, out2, mod2_label)

                val_metrics["loss"] += float(batch_loss) / max(1, len(self.loader_val))
                val_metrics["loss_mod1"] += float(loss1) / max(1, len(self.loader_val))
                val_metrics["loss_mod2"] += float(loss2) / max(1, len(self.loader_val))

        return val_metrics, metrics_mod1.compute_stats(), metrics_mod2.compute_stats()

    def _update_metrics(self, metric_calculator, pred, target):
        values = metric_calculator(pred, target)
        metric_calculator.update(*values)

    def _log_segmentation_epoch(self, epoch, train_metrics, val_metrics, val_stats_mod1, val_stats_mod2):
        self.logger.info(
            f"Epoch {epoch}/{self.config.nb_epochs} - "
            f"train_loss={train_metrics['loss']:.4f}, val_loss={val_metrics['loss']:.4f}, "
            f"val_dice_mod1={val_stats_mod1['dice']['str']}, val_dice_mod2={val_stats_mod2['dice']['str']}, "
            f"best_val_dice={self.best_val_dice:.4f}"
        )

        if self.config.data == "BraTs19":
            self.logger.info(
                "BraTS regions - "
                f"Mod1 ET={val_stats_mod1['dice']['regions']['ET']['str']}, "
                f"TC={val_stats_mod1['dice']['regions']['TC']['str']}, "
                f"WT={val_stats_mod1['dice']['regions']['WT']['str']}; "
                f"Mod2 ET={val_stats_mod2['dice']['regions']['ET']['str']}, "
                f"TC={val_stats_mod2['dice']['regions']['TC']['str']}, "
                f"WT={val_stats_mod2['dice']['regions']['WT']['str']}"
            )

    def load_model(self, path):
        return self._load_weights(path, strict=True, load_optimizer=True)

    def _load_weights(self, path, strict, load_optimizer):
        try:
            checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        except BaseException as exc:
            raise ValueError(f"Unable to load checkpoint {path}: {exc}") from exc

        if hasattr(checkpoint, "state_dict"):
            self.model.load_state_dict(checkpoint.state_dict(), strict=strict)
            return 1

        if not isinstance(checkpoint, dict):
            self.model.load_state_dict(checkpoint, strict=strict)
            return 1

        if "model" not in checkpoint:
            raise ValueError(f"Checkpoint {path} does not contain a 'model' state dict.")

        info = self.model.load_state_dict(checkpoint["model"], strict=strict)
        self.logger.info(f"Loaded model: {info}")

        if "loss" in checkpoint and checkpoint["loss"] is not None and hasattr(self.loss, "load_state_dict"):
            self.loss.load_state_dict(checkpoint["loss"])
        if load_optimizer and self.optimizer is not None and checkpoint.get("optimizer") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if load_optimizer and self.scheduler is not None and checkpoint.get("scheduler") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        return int(checkpoint.get("epoch", 0)) + 1
