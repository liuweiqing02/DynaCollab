# SSCLseg

SSCLseg 是一个面向 3D 医学图像的双模态分割训练项目，支持预训练与微调两阶段流程，提供统一训练入口、自动化实验目录管理、断点恢复与早停机制。

## 1. 功能概览

- 双模态 3D 分割训练（如 CT + MR）。
- 两阶段训练流程：`pretraining` 与 `finetuning`。
- 多模型支持：`CrossModalUNet`、`UNet`、`VTUNet`、`UNETR`、`UniUnet`。
- 多融合策略支持：`baseline`、`daa`、`daa_cmau`。
- 自动保存运行日志、最佳模型、最新模型、配置快照。
- 支持断点恢复与可配置早停。

## 2. 项目结构

- `main.py`：统一训练入口（推荐）。
- `config.py`：全局配置与实验命名逻辑。
- `training/runtime.py`：训练流程编排。
- `training/builders.py`：数据加载器、模型、损失构建。
- `dataset.py`：数据读取、预处理、增强、划分。
- `augmentations.py`：增强算子实现。
- `CompatibleModel.py`：核心训练循环与验证逻辑。
- `models/`：模型定义。
- `testdata.py`：推理与结果导出脚本。
- `finetune_brats_fused.py`：BraTS 融合微调脚本（可选）。
- `finetune_dongmai_singlemod.py`：Dongmai 单模态微调脚本（可选）。

## 3. 环境要求

- Python 3.8 及以上。
- 推荐使用 CUDA + PyTorch GPU 训练环境。

常用依赖（按需安装）：
- `torch`
- `numpy`
- `scipy`
- `scikit-image`
- `nibabel`
- `pandas`
- `openpyxl`
- `scikit-learn`
- `tqdm`
- `matplotlib`
- `seaborn`

示例安装命令：

```bash
pip install torch torchvision torchaudio
pip install numpy scipy scikit-image nibabel pandas openpyxl scikit-learn tqdm matplotlib seaborn
```

## 4. 数据准备

在 `config.py` 中配置数据路径。

### 4.1 dongmai

需要配置以下目录：
- `image_dir_mod1_tr`
- `label_dir_mod1_tr`
- `image_dir_mod2_tr`
- `label_dir_mod2_tr`

要求：
- 两个模态与标签文件名中的病例 ID 一一对应。
- 文件格式为 NIfTI（`.nii.gz`）。

### 4.2 BraTs19

需要配置：
- `dir_tr`

默认读取每个病例目录中的：
- `<prefix>_t1ce.nii`
- `<prefix>_flair.nii`
- `<prefix>_seg.nii`

## 5. 快速开始

### 5.1 预训练

```bash
python main.py --mode pretraining
```

### 5.2 微调（加载预训练 best）

```bash
python main.py --mode finetuning --pretrained_path ./runs/<pretrain_run>/checkpoints/best.pth
```

### 5.3 临时覆盖增强策略

```bash
python main.py --mode pretraining --tf all_tf
python main.py --mode finetuning --tf no_tf
```

### 5.4 临时覆盖训练/验证划分

```bash
python main.py --mode pretraining --train_ratio 0.8 --split_seed 42
```

## 6. 配置说明（`config.py`）

常用参数如下：

| 参数 | 含义 |
|---|---|
| `data` | 数据集选择（`dongmai` / `BraTs19`） |
| `model` | 模型名称 |
| `fusion_strategy` | 融合策略（`baseline` / `daa` / `daa_cmau`） |
| `use_global_local_loss` | 预训练是否使用全局+局部损失 |
| `batch_size` / `batch_size_val` | 训练/验证 batch 大小 |
| `lr` | 学习率 |
| `weight_decay` | 权重衰减 |
| `nb_epochs` | 训练轮数 |
| `tf` | 增强开关（`no_tf` / `all_tf`） |
| `desired_spacing` | 重采样 spacing |
| `target_size` / `target_size_crop` | pad/crop 目标尺寸 |
| `enable_memory_cache` | 是否缓存预处理后的样本 |
| `train_ratio` / `split_seed` | 训练验证划分比例与随机种子 |
| `use_early_stopping` | 是否启用早停 |
| `early_stopping_patience` | 早停耐心轮数 |
| `early_stopping_min_delta` | 指标最小改善阈值 |
| `early_stopping_start_epoch` | 从哪一轮开始早停判断 |

## 7. 中断与恢复

训练过程中可使用 `Ctrl+C` 安全中断。项目每个 epoch 会保存：
- `latest.pth`：最新 checkpoint。
- `best.pth`：当前最佳 checkpoint。

### 7.1 恢复预训练

```bash
python main.py --mode pretraining --resume_checkpoint ./runs/<run_name>/checkpoints/latest.pth
```

### 7.2 恢复微调

```bash
python main.py --mode finetuning --resume_checkpoint ./runs/<run_name>/checkpoints/latest.pth
```

说明：
- `--resume_checkpoint` 会恢复模型参数、优化器与调度器状态。
- `--pretrained_path` 仅加载模型权重，适合启动新的微调任务。

## 8. 推理与结果导出

可使用 `testdata.py` 对训练后模型进行推理并导出 NIfTI 结果：

```bash
python testdata.py --model_path ./runs/<run_name>/checkpoints/best.pth --config dongmai
python testdata.py --model_path ./runs/<run_name>/checkpoints/best.pth --config BraTs19
```

输出目录：
- `test_result/<model_name>/`

## 9. 运行产物

每次训练会在 `runs/<run_name>/` 下生成：
- `logs/training.log`
- `logs/validation_ids.csv`
- `checkpoints/latest.pth`
- `checkpoints/best.pth`
- `config_snapshot.json`

## 10. 调参建议

推荐初始方案：
- 预训练使用 `tf=all_tf`。
- 微调使用 `tf=no_tf`。
- 固定 `train_ratio=0.8` 与 `split_seed=42` 保证可复现。

推荐调参顺序：
1. 学习率 `lr`
2. 批大小 `batch_size`
3. 训练轮数 `nb_epochs`
4. 早停参数 `patience` 与 `min_delta`
5. 融合策略与网络规模（`fusion_strategy`、`growth_rate`）

