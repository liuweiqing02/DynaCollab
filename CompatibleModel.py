import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn import DataParallel
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, ChainedScheduler
from tqdm import tqdm
import logging
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

from Visualizer import SegmentationVisualizer, PhaseVisualizer
from metrics import SegmentationMetrics, BraTSMetrics


class CompatibleModel:

    def __init__(self, net_1, net_2, loss, loader_train, loader_val, config, dataset_val=None, scheduler=None):
        """

        Parameters
        ----------
        net: subclass of nn.Module
        loss: callable fn with args (y_pred, y_true)
        loader_train, loader_val: pytorch DataLoaders for training/validation
        config: Config object with hyperparameters
        scheduler (optional)
        """
        super().__init__()
        self.loss = loss
        self.model1 = net_1
        self.model2 = net_2
        self.scheduler = scheduler
        self.loader = loader_train
        self.loader_val = loader_val
        self.device = torch.device("cuda" if config.cuda else "cpu")
        if config.cuda and not torch.cuda.is_available():
            raise ValueError("No GPU found: set cuda=False parameter.")
        self.config = config
        self.best_val_dice = 0.0  # 跟踪最佳验证准确率
        self.best_val_loss = 100
        self.logger = logging.getLogger("CLseg")

        self.is_single_stream = (net_2 is None)

        # 先DataParallel再load_model，不然FT的时候模型文件参数对不上，unexpected会返回一堆参数没对齐
        self.model1 = DataParallel(self.model1).to(self.device)
        if not self.is_single_stream:
            self.model2 = DataParallel(self.model2).to(self.device)

        self.dataset_val = dataset_val
        self.visualizer3D = SegmentationVisualizer(config)

    def _latest_ckpt_path(self):
        if hasattr(self.config, "latest_checkpoint_path"):
            return self.config.latest_checkpoint_path
        return os.path.join(self.config.checkpoint_dir, f"{self.config.checkpoint_name}_latest.pth")

    def _best_ckpt_path(self):
        if hasattr(self.config, "best_checkpoint_path"):
            return self.config.best_checkpoint_path
        return os.path.join(self.config.checkpoint_dir, f"{self.config.checkpoint_name}_best.pth")

    def _early_stopping_cfg(self):
        enabled = bool(getattr(self.config, "use_early_stopping", False))
        patience = int(getattr(self.config, "early_stopping_patience", 0))
        min_delta = float(getattr(self.config, "early_stopping_min_delta", 0.0))
        start_epoch = int(getattr(self.config, "early_stopping_start_epoch", 1))
        return {
            "enabled": enabled and patience > 0,
            "patience": max(patience, 0),
            "min_delta": max(min_delta, 0.0),
            "start_epoch": max(start_epoch, 1),
        }

    # def _init_visualizer(self, phase, resume):
    #     self.visualizer = PhaseVisualizer(
    #         self.config,
    #         phase=phase,
    #         resume=resume
    #     )

    def pretraining(self):
        start_epoch = 1
        # self._init_visualizer('pretraining', resume=True)

        if self.is_single_stream:
            # 单流模型 - 只有 model1 的参数
            model_params = list(self.model1.parameters())
        else:
            # 双流模型 - 两个模型的参数
            model_params = list(self.model1.parameters()) + list(self.model2.parameters())
        # temperature_params = self.loss.temperature  # 温度参数单独分组
        # 配置不同参数组的学习率
        optimizer_groups = [
            {  # 模型参数组（较低学习率）
                "params": model_params,
                "lr": self.config.lr,  # 例如 3e-4
                "weight_decay": self.config.weight_decay
            },
            # {  # 温度参数组（较高学习率）
            #     "params": temperature_params,
            #     "lr": self.config.lr * 0.2,  # 温度参数的学习率（通常比模型高10~100倍）
            #     "weight_decay": self.config.weight_decay  # 可选：不应用权重衰减
            # }
        ]
        self.optimizer = torch.optim.AdamW(optimizer_groups)

        # 线性预热（从 1% 到 100% 学习率）
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,  # 初始学习率 = base_lr * 0.01
            end_factor=1.0,  # 预热结束学习率 = base_lr * 1.0
            total_iters=10
        )

        # 余弦退火（预热后生效）
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.nb_epochs - 10)

        # 链式调度器
        self.scheduler = ChainedScheduler([warmup_scheduler, cosine_scheduler])

        print(self.loss)
        print(self.optimizer)

        if hasattr(self.config, 'pretrained_checkpoint_path') and self.config.pretrained_checkpoint_path is not None:
            start_epoch = self.load_model(self.config.pretrained_checkpoint_path)
        print('start_epoch=', start_epoch)

        early_cfg = self._early_stopping_cfg()
        early_stop_bad_epochs = 0
        if early_cfg["enabled"]:
            self.logger.info(
                f"Early stopping enabled (pretraining): "
                f"patience={early_cfg['patience']}, min_delta={early_cfg['min_delta']}, "
                f"start_epoch={early_cfg['start_epoch']}"
            )

        for epoch in range(start_epoch, self.config.nb_epochs + 1):

            ## Training step
            self.model1.train()
            if not self.is_single_stream:
                self.model2.train()
            nb_batch = len(self.loader)
            training_loss = 0
            pbar = tqdm(total=nb_batch, desc="Training")

            for batch_idx, batch in enumerate(self.loader):
                mod1, mod2, mod1_label, mod2_label, id = batch
                pbar.update()
                mod1 = mod1.to(self.device)
                mod2 = mod2.to(self.device)
                mod1_label = mod1_label.to(self.device)
                mod2_label = mod2_label.to(self.device)
                self.optimizer.zero_grad()
                # 根据模型类型进行前向传播
                if self.is_single_stream:
                    if self.config.use_global_local_loss:
                        global_projs, local_projs, deform_params = self.model1([mod1, mod2])
                        z_global_i, z_global_j = global_projs
                        z_local_i, z_local_j = local_projs
                    else:
                        # 单流模型：同时输入两个模态
                        projections = self.model1([mod1, mod2])
                        # 确保返回两个投影（用于对比学习）
                        if len(projections) != 2:
                            raise ValueError("单流模型应返回两个模态的投影")
                        z_i, z_j = projections
                else:
                    # 双流模型：每个模型处理一个模态
                    z_i = self.model1(mod1)
                    z_j = self.model2(mod2)
                if self.config.use_global_local_loss:
                    # 计算联合损失
                    batch_loss = self.loss(
                        z_global_i, z_global_j,
                        z_local_i, z_local_j,
                        mod1_label, mod2_label, deform_params_list=deform_params
                    )
                    # 检查全局和局部特征是否包含NaN
                    if torch.isnan(z_global_i).any() or torch.isnan(z_global_j).any() or torch.isnan(z_local_i).any() or torch.isnan(z_local_j).any():
                        self.logger.warning("NaN特征值检测到，跳过该batch")
                        self.optimizer.zero_grad(set_to_none=True)
                        # 强制释放所有中间变量
                        del mod1, mod2, mod1_label, mod2_label, global_projs, local_projs
                        del z_global_i, z_global_j, z_local_i, z_local_j
                        torch.cuda.empty_cache()  # 关键：清空GPU缓存
                        pbar.update()
                        continue  # 跳过当前batch
                else:
                    batch_loss = self.loss(z_i, z_j)
                batch_loss.backward()

                torch.nn.utils.clip_grad_norm_(model_params, max_norm=1.0)  # 梯度裁剪

                self.optimizer.step()
                training_loss += float(batch_loss) / nb_batch

            pbar.close()

            ## Validation step
            nb_batch = len(self.loader_val)
            pbar = tqdm(total=nb_batch, desc="Validation")
            val_loss = 0
            val_values = {}
            with torch.no_grad():
                self.model1.eval()
                if not self.is_single_stream:
                    self.model2.eval()
                for batch in self.loader_val:
                    mod1, mod2, mod1_label, mod2_label, id = batch
                    pbar.update()
                    mod1 = mod1.to(self.device)
                    mod2 = mod2.to(self.device)
                    mod1_label = mod1_label.to(self.device)
                    mod2_label = mod2_label.to(self.device)
                    # 根据模型类型进行前向传播
                    if self.is_single_stream:
                        if self.config.use_global_local_loss:
                            global_projs, local_projs, deform_params = self.model1([mod1, mod2])
                            z_global_i, z_global_j = global_projs
                            z_local_i, z_local_j = local_projs
                        else:
                            projections= self.model1([mod1, mod2])
                            z_i, z_j = projections
                    else:
                        z_i = self.model1(mod1)
                        z_j = self.model2(mod2)
                    if self.config.use_global_local_loss:
                        # 计算联合损失
                        batch_loss = self.loss(
                            z_global_i, z_global_j,
                            z_local_i, z_local_j,
                            mod1_label, mod2_label, deform_params_list=deform_params
                        )
                    else:
                        batch_loss = self.loss(z_i, z_j)
                    val_loss += float(batch_loss) / nb_batch
            pbar.close()

            if self.scheduler is not None:
                self.scheduler.step()

            self.logger.info(
                f"Epoch [{epoch}/{self.config.nb_epochs}] "
                f"Training loss = {training_loss:.4f}\t"
                f"Validation loss = {val_loss:.4f}"
            )

            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            checkpoint = {
                "epoch": epoch,
                "model1": self.model1.state_dict(),
                "loss": self.loss.state_dict() if hasattr(self.loss, 'state_dict') else None,
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "optimizer": self.optimizer.state_dict()
            }

            # 如果是双流模型，保存第二个模型
            if not self.is_single_stream:
                checkpoint["model2"] = self.model2.state_dict()

            torch.save(checkpoint, self._latest_ckpt_path())
            current_val_loss = val_loss
            is_best = current_val_loss < (self.best_val_loss - early_cfg["min_delta"])
            if is_best:
                self.best_val_loss = current_val_loss
                torch.save(checkpoint, self._best_ckpt_path())
                early_stop_bad_epochs = 0
                self.logger.info(f"🌟 New Best Val Loss: {val_loss:.4f} at Epoch {epoch}")
            elif early_cfg["enabled"] and epoch >= early_cfg["start_epoch"]:
                early_stop_bad_epochs += 1
                self.logger.info(
                    f"Early stopping wait (pretraining): "
                    f"{early_stop_bad_epochs}/{early_cfg['patience']}"
                )
                if early_stop_bad_epochs >= early_cfg["patience"]:
                    self.logger.info(
                        f"Early stopping triggered at epoch {epoch} "
                        f"(best_val_loss={self.best_val_loss:.4f})"
                    )
                    break

            # self.visualizer.update_metrics(epoch,
            #                                loss=training_loss,
            #                                val_loss=val_loss)
        self.logger.info(
            "\n=== Pretraining Completed ==="
            f"\nBest Validation Loss: {self.best_val_loss} "
        )

    def fine_tuning(self):
        # del self.model1
        start_epoch = 1
        # self._init_visualizer('finetuning', resume=True)
        if self.is_single_stream:
            # 单流模型 - 只有 model1 的参数
            model_params = list(self.model1.parameters())
        else:
            # 双流模型 - 两个模型的参数
            model_params = list(self.model1.parameters()) + list(self.model2.parameters())

        self.optimizer = torch.optim.AdamW(model_params, lr=self.config.lr, weight_decay=self.config.weight_decay)

        # 线性预热（从 1% 到 100% 学习率）
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,  # 初始学习率 = base_lr * 0.01
            end_factor=1.0,  # 预热结束学习率 = base_lr * 1.0
            total_iters=10
        )

        # 余弦退火（预热后生效）
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.nb_epochs - 10)

        # 链式调度器
        self.scheduler = ChainedScheduler([warmup_scheduler, cosine_scheduler])

        # 初始化指标计算器
        self.metrics_calculator = SegmentationMetrics(self.config.num_classes,spacing=self.config.desired_spacing)

        print("=== 优化器参数组 ===")
        for idx, group in enumerate(self.optimizer.param_groups):
            print(f"Group {idx}: LR={group['lr']}, Parameters={len(group['params'])}")

        print(self.loss)
        print(self.optimizer)
        # 加载预训练模型，不需要加载优化器
        if hasattr(self.config, 'pretrained_path') and self.config.pretrained_path is not None:
            checkpoint = None
            try:
                checkpoint = torch.load(self.config.pretrained_path,
                                        map_location=lambda storage, loc: storage)
            except BaseException as e:
                self.logger.error('Impossible to load the checkpoint: %s' % str(e))
            try:
                if "model1" in checkpoint:
                    unexpected = self.model1.load_state_dict(checkpoint["model1"], strict=False)
                    self.logger.info('Model1 loading info: {}'.format(unexpected))
                if "model2" in checkpoint:
                    unexpected = self.model2.load_state_dict(checkpoint["model2"], strict=False)
                    self.logger.info('Model2 loading info: {}'.format(unexpected))
            except BaseException as e:
                raise ValueError('Error while loading the model\'s weights: %s' % str(e))


        # 加载微调训练中断时的模型，需要加载优化器
        elif hasattr(self.config,
                     'finetuning_checkpoint_path') and self.config.finetuning_checkpoint_path is not None:
            start_epoch = self.load_model(self.config.finetuning_checkpoint_path)

        # 初始化指标计算器
        if self.config.data=='BraTs19':
            metrics_calculator_mod1 = BraTSMetrics(spacing=self.config.desired_spacing)
            metrics_calculator_mod2 = BraTSMetrics(spacing=self.config.desired_spacing)
        else:
            metrics_calculator_mod1 = SegmentationMetrics(self.config.num_classes,spacing=self.config.desired_spacing)
            metrics_calculator_mod2 = SegmentationMetrics(self.config.num_classes,spacing=self.config.desired_spacing)

        skip_metrics_epochs = 40
        early_cfg = self._early_stopping_cfg()
        early_cfg["start_epoch"] = max(early_cfg["start_epoch"], skip_metrics_epochs + 1)
        early_stop_bad_epochs = 0
        if early_cfg["enabled"]:
            self.logger.info(
                f"Early stopping enabled (finetuning): "
                f"patience={early_cfg['patience']}, min_delta={early_cfg['min_delta']}, "
                f"start_epoch={early_cfg['start_epoch']}"
            )

        for epoch in range(start_epoch, self.config.nb_epochs + 1):
            skip_metrics = (epoch <= skip_metrics_epochs)
            # 重置指标收集器
            metrics_calculator_mod1.reset()
            metrics_calculator_mod2.reset()
            ## Training step
            self.model1.train()
            if not self.is_single_stream:
                self.model2.train()
            nb_batch = len(self.loader)
            pbar = tqdm(total=nb_batch, desc="Training")
            train_metrics = {'loss': 0.0, 'loss_mod1': 0.0, 'loss_mod2': 0.0}
            for batch in self.loader:
                mod1, mod2, mod1_label, mod2_label, id = batch
                mod1, mod2 = mod1.to(self.device), mod2.to(self.device)
                # 调整标签维度 [B,1,D,H,W] -> [B,D,H,W]
                mod1_label = mod1_label.squeeze(1).long().to(self.device)
                mod2_label = mod2_label.squeeze(1).long().to(self.device)
                pbar.update()

                self.optimizer.zero_grad()
                if self.is_single_stream:
                    # 单流模型：同时输入两个模态
                    outputs = self.model1([mod1, mod2])
                    # 确保返回两个投影（用于对比学习）
                    if len(outputs) != 2:
                        raise ValueError("单流模型应返回两个模态的投影")
                    out1, out2 = outputs
                else:
                    # 双流模型：每个模型处理一个模态
                    out1 = self.model1(mod1)  # [B, C, D, H, W]
                    out2 = self.model2(mod2)
                # 计算双流损失
                loss1 = self.loss(out1, mod1_label)
                loss2 = self.loss(out2, mod2_label)
                batch_loss = loss1 + loss2

                batch_loss.backward()
                self.optimizer.step()

                # 计算指标
                if not skip_metrics:
                    with torch.no_grad():
                        dice1, iou1, hd95_1 = metrics_calculator_mod1(out1, mod1_label)
                        dice2, iou2, hd95_2 = metrics_calculator_mod2(out2, mod2_label)
                        metrics_calculator_mod1.update(dice1, iou1, hd95_1)
                        metrics_calculator_mod2.update(dice2, iou2, hd95_2)

                # 累计统计量
                train_metrics['loss'] += batch_loss.item() / nb_batch
                train_metrics['loss_mod1'] += loss1.item() / nb_batch
                train_metrics['loss_mod2'] += loss2.item() / nb_batch

            pbar.close()

            ## Validation step
            nb_batch = len(self.loader_val)
            pbar = tqdm(total=nb_batch, desc="Validation")
            val_metrics = {'loss': 0.0, 'loss_mod1': 0.0, 'loss_mod2': 0.0}
            with torch.no_grad():
                # 为验证集创建新的指标收集器
                if self.config.data == 'BraTs19':
                    val_metrics_mod1 = BraTSMetrics(spacing=self.config.desired_spacing)
                    val_metrics_mod2 = BraTSMetrics(spacing=self.config.desired_spacing)
                else:
                    # 修改：添加spacing参数
                    val_metrics_mod1 = SegmentationMetrics(
                        self.config.num_classes,
                        spacing=self.config.desired_spacing
                    )
                    val_metrics_mod2 = SegmentationMetrics(
                        self.config.num_classes,
                        spacing=self.config.desired_spacing
                    )
                self.model1.eval()
                if not self.is_single_stream:
                    self.model2.eval()
                for batch in self.loader_val:
                    mod1, mod2, mod1_label, mod2_label, id = batch
                    mod1, mod2 = mod1.to(self.device), mod2.to(self.device)
                    mod1_label = mod1_label.squeeze(1).long().to(self.device)
                    mod2_label = mod2_label.squeeze(1).long().to(self.device)
                    pbar.update()

                    if self.is_single_stream:
                        outputs = self.model1([mod1, mod2])
                        out1, out2 = outputs
                    else:
                        # 双流模型：每个模型处理一个模态，不需要做隔离
                        out1 = self.model1(mod1)  # [B, C, D, H, W]
                        out2 = self.model2(mod2)

                    # 计算验证损失
                    loss1 = self.loss(out1, mod1_label)
                    loss2 = self.loss(out2, mod2_label)
                    batch_loss = loss1 + loss2

                    # 计算验证指标
                    # 只在50个epoch后计算验证指标
                    if not skip_metrics:
                        dice1, iou1, hd95_1 = val_metrics_mod1(out1, mod1_label)
                        dice2, iou2, hd95_2 = val_metrics_mod2(out2, mod2_label)
                        val_metrics_mod1.update(dice1, iou1, hd95_1)
                        val_metrics_mod2.update(dice2, iou2, hd95_2)
                    # 累计统计量
                    val_metrics['loss'] += batch_loss.item() / len(self.loader_val)
                    val_metrics['loss_mod1'] += loss1.item() / len(self.loader_val)
                    val_metrics['loss_mod2'] += loss2.item() / len(self.loader_val)


            pbar.close()

            # 计算统计信息
            if not skip_metrics:
                train_stats_mod1 = metrics_calculator_mod1.compute_stats()
                train_stats_mod2 = metrics_calculator_mod2.compute_stats()
                val_stats_mod1 = val_metrics_mod1.compute_stats()
                val_stats_mod2 = val_metrics_mod2.compute_stats()


            # 打印指标
            if skip_metrics:
                # 前50个epoch只打印损失
                log_msg = (
                    f"Epoch [{epoch}/{self.config.nb_epochs}] (Skipping metrics)\n"
                    f"Train Loss: {train_metrics['loss']:.4f} "
                    f"(mod1: {train_metrics['loss_mod1']:.4f}, mod2: {train_metrics['loss_mod2']:.4f})\n"
                    f"Val Loss: {val_metrics['loss']:.4f} "
                    f"(mod1: {val_metrics['loss_mod1']:.4f}, mod2: {val_metrics['loss_mod2']:.4f})"
                )
            else:
                log_msg = (
                    f"Epoch [{epoch}/{self.config.nb_epochs}]\n"
                    f"Train Loss: {train_metrics['loss']:.4f} "
                    f"(mod1: {train_metrics['loss_mod1']:.4f}, mod2: {train_metrics['loss_mod2']:.4f})\n"
                    f"Val Loss: {val_metrics['loss']:.4f} "
                    f"(mod1: {val_metrics['loss_mod1']:.4f}, mod2: {val_metrics['loss_mod2']:.4f})\n"
                    f"Train Dice: Mod1 {train_stats_mod1['dice']['str']}, Mod2 {train_stats_mod2['dice']['str']}\n"
                    f"Val Dice: Mod1 {val_stats_mod1['dice']['str']}, Mod2 {val_stats_mod2['dice']['str']}\n"
                    f"Train IoU: Mod1 {train_stats_mod1['iou']['str']}, Mod2 {train_stats_mod2['iou']['str']}\n"
                    f"Val IoU: Mod1 {val_stats_mod1['iou']['str']}, Mod2 {val_stats_mod2['iou']['str']}\n"
                    # 新增：HD95指标（所有数据集）
                    f"Train HD95: Mod1 {train_stats_mod1['hd95']['str']}, Mod2 {train_stats_mod2['hd95']['str']}\n"
                    f"Val HD95: Mod1 {val_stats_mod1['hd95']['str']}, Mod2 {val_stats_mod2['hd95']['str']}\n"
                )

                # 添加区域指标（如果是BraTS19）
                if self.config.data == 'BraTs19':
                    # ... [原有BraTS区域指标代码保持不变] ...
                    # Mod1 训练指标
                    log_msg += f"Train Mod1 - ET: {train_stats_mod1['dice']['regions']['ET']['str']}, " \
                               f"TC: {train_stats_mod1['dice']['regions']['TC']['str']}, " \
                               f"WT: {train_stats_mod1['dice']['regions']['WT']['str']}\n"
                    # Mod2 训练指标
                    log_msg += f"Train Mod2 - ET: {train_stats_mod2['dice']['regions']['ET']['str']}, " \
                               f"TC: {train_stats_mod2['dice']['regions']['TC']['str']}, " \
                               f"WT: {train_stats_mod2['dice']['regions']['WT']['str']}\n"
                    # Mod1 验证指标
                    log_msg += f"Val Mod1 - ET: {val_stats_mod1['dice']['regions']['ET']['str']}, " \
                               f"TC: {val_stats_mod1['dice']['regions']['TC']['str']}, " \
                               f"WT: {val_stats_mod1['dice']['regions']['WT']['str']}\n"
                    # Mod2 验证指标
                    log_msg += f"Val Mod2 - ET: {val_stats_mod2['dice']['regions']['ET']['str']}, " \
                               f"TC: {val_stats_mod2['dice']['regions']['TC']['str']}, " \
                               f"WT: {val_stats_mod2['dice']['regions']['WT']['str']}\n"
                    # 新增：HD95区域指标
                    log_msg += f"Val Mod1 HD95 - ET: {val_stats_mod1['hd95']['regions']['ET']['str']}, " \
                               f"TC: {val_stats_mod1['hd95']['regions']['TC']['str']}, " \
                               f"WT: {val_stats_mod1['hd95']['regions']['WT']['str']}\n"
                    log_msg += f"Val Mod2 HD95 - ET: {val_stats_mod2['hd95']['regions']['ET']['str']}, " \
                               f"TC: {val_stats_mod2['hd95']['regions']['TC']['str']}, " \
                               f"WT: {val_stats_mod2['hd95']['regions']['WT']['str']}\n"
            self.logger.info(log_msg)

            # # 更新可视化工具
            # if skip_metrics:
            #     # 前50个epoch只更新损失
            #     self.visualizer.update_metrics(
            #         epoch,
            #         train_loss=train_metrics['loss'],
            #         train_loss_mod1=train_metrics['loss_mod1'],
            #         train_loss_mod2=train_metrics['loss_mod2'],
            #         val_loss=val_metrics['loss'],
            #         val_loss_mod1=val_metrics['loss_mod1'],
            #         val_loss_mod2=val_metrics['loss_mod2'],
            #     )
            # else:
            #     # 50个epoch后更新所有指标
            #     self.visualizer.update_metrics(
            #         epoch,
            #         train_loss=train_metrics['loss'],
            #         train_loss_mod1=train_metrics['loss_mod1'],
            #         train_loss_mod2=train_metrics['loss_mod2'],
            #         train_dice_mod1=train_stats_mod1['dice']['mean'],
            #         train_dice_mod2=train_stats_mod2['dice']['mean'],
            #         train_iou_mod1=train_stats_mod1['iou']['mean'],
            #         train_iou_mod2=train_stats_mod2['iou']['mean'],
            #         val_loss=val_metrics['loss'],
            #         val_loss_mod1=val_metrics['loss_mod1'],
            #         val_loss_mod2=val_metrics['loss_mod2'],
            #         val_dice_mod1=val_stats_mod1['dice']['mean'],
            #         val_dice_mod2=val_stats_mod2['dice']['mean'],
            #         val_iou_mod1=val_stats_mod1['iou']['mean'],
            #         val_iou_mod2=val_stats_mod2['iou']['mean']
            #     )

            if self.scheduler is not None:
                self.scheduler.step()

            # ==== 保存模型检查点 ====
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            checkpoint = {
                "epoch": epoch,
                "model1": self.model1.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            }
            # 如果是双流模型，保存第二个模型
            if not self.is_single_stream:
                checkpoint["model2"] = self.model2.state_dict()
            # 保存最新
            torch.save(checkpoint, self._latest_ckpt_path())

            # 可视化样本（只在50个epoch后且是最佳模型时）
            if not skip_metrics:
                # self.visualize_samples(epoch, is_best=False)
                current_val_dice = (val_stats_mod1['dice']['mean'] + val_stats_mod2['dice']['mean']) / 2
                is_best = current_val_dice > (self.best_val_dice + early_cfg["min_delta"])
                if is_best:
                    self.best_val_dice = current_val_dice
                    torch.save(checkpoint, self._best_ckpt_path())
                    early_stop_bad_epochs = 0
                    # self.visualize_samples(epoch, is_best=True)
                    self.logger.info(
                        f"🌟 New Best Val Dice: {current_val_dice:.4f} "
                        f"(Mod1: {val_stats_mod1['dice']['mean']:.4f}, Mod2: {val_stats_mod2['dice']['mean']:.4f})"
                    )

                elif early_cfg["enabled"] and epoch >= early_cfg["start_epoch"]:
                    early_stop_bad_epochs += 1
                    self.logger.info(
                        f"Early stopping wait (finetuning): "
                        f"{early_stop_bad_epochs}/{early_cfg['patience']}"
                    )
                    if early_stop_bad_epochs >= early_cfg["patience"]:
                        self.logger.info(
                            f"Early stopping triggered at epoch {epoch} "
                            f"(best_val_dice={self.best_val_dice:.4f})"
                        )
                        break

    def load_model(self, path):
        checkpoint = None
        try:
            checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        except BaseException as e:
            self.logger.error('Impossible to load the checkpoint: %s' % str(e))
        if checkpoint is not None:
            try:
                if hasattr(checkpoint, "state_dict"):
                    unexpected1 = self.model1.load_state_dict(checkpoint.state_dict())
                    self.logger.info('Model1 loading info: {}'.format(unexpected1))
                    unexpected2 = self.model2.load_state_dict(checkpoint.state_dict())
                    self.logger.info('Model2 loading info: {}'.format(unexpected2))
                elif isinstance(checkpoint, dict):
                    if "model1" in checkpoint:
                        unexpected1 = self.model1.load_state_dict(
                            checkpoint["model1"])  # 去掉strict=False，必须完全匹配*********************8
                        self.logger.info('Model1 loading info: {}'.format(unexpected1))
                    if "model2" in checkpoint:
                        unexpected2 = self.model2.load_state_dict(
                            checkpoint["model2"])  # 去掉strict=False，必须完全匹配*********************8
                        self.logger.info('Model2 loading info: {}'.format(unexpected2))
                    if "loss" in checkpoint:
                        unexpected3 = self.loss.load_state_dict(checkpoint["loss"])  # 新增此行
                        self.logger.info('loss loading info: {}'.format(unexpected3))
                    if "scheduler" in checkpoint:
                        self.scheduler.load_state_dict(checkpoint["scheduler"])
                    if "epoch" in checkpoint:
                        self.optimizer.load_state_dict(checkpoint['optimizer'])  # 优化器的参数也要加载
                        return checkpoint["epoch"] + 1
                else:
                    unexpected1 = self.model1.load_state_dict(checkpoint)
                    self.logger.info('Model1 loading info: {}'.format(unexpected1))
                    unexpected2 = self.model2.load_state_dict(checkpoint)
                    self.logger.info('Model2 loading info: {}'.format(unexpected2))
            except BaseException as e:
                raise ValueError('Error while loading the model\'s weights: %s' % str(e))

    def visualize_samples(self, epoch, is_best=False):
        """更新后的可视化入口"""
        if not hasattr(self, 'dataset_val') or len(self.dataset_val) == 0:
            return

        sample_idx = 3
        self.visualizer3D.visualize_samples(
            dataset=self.dataset_val,
            model=(self.model1, self.model2),
            device=self.device,
            epoch=epoch,
            is_best=is_best,
            case_idx=sample_idx
        )
