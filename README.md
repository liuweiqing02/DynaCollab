# DynaCollab

Official implementation of **DynaCollab**, a dual-modal 3D medical image segmentation framework with dynamic cross-modal collaboration.

DynaCollab combines a dual-modal segmentation backbone with **Dynamic Anatomical Alignment (DAA)** and **Cross-Modality Alignment Units (CMAU)**. For pretraining, the code supports the proposed **Two-Stage Task-Aware Collaborative Contrastive Loss (TSTCL)** as well as a standard dual-modal contrastive objective.

## Installation

```bash
conda create -n dynacollab python=3.9
conda activate dynacollab
pip install -r requirements.txt
```

## Data Preparation

Create a local `.env` file from the template and set the dataset locations:

```bash
cp .env.example .env
```

For carotid CT/MRI experiments, configure:

```bash
DYNACOLLAB_CAROTID_CT_TRAIN_IMAGES=./data/CarotidArtery_CT/imagesTr
DYNACOLLAB_CAROTID_CT_TRAIN_LABELS=./data/CarotidArtery_CT/labelsTr
DYNACOLLAB_CAROTID_MRI_TRAIN_IMAGES=./data/CarotidArtery_MRI/imagesTr
DYNACOLLAB_CAROTID_MRI_TRAIN_LABELS=./data/CarotidArtery_MRI/labelsTr
```

For BraTS19 experiments, configure:

```bash
DYNACOLLAB_BRATS_TRAIN_DIR=./data/BraTS19/HGG
```

The training code reads NIfTI files (`.nii` or `.nii.gz`) and performs resampling, center cropping, paired augmentation, train/validation splitting, checkpointing, and metric logging.

## Quick Start

Pretrain DynaCollab with the default setting:

```bash
python main.py --mode pretraining --data carotid
```

Fine-tune from a pretrained checkpoint:

```bash
python main.py --mode finetuning --data carotid --pretrained_path ./runs/<run>/checkpoints/best.pth
```

Use BraTS19 by changing the dataset argument:

```bash
python main.py --mode pretraining --data BraTs19
```

## Reproducible Configurations

The main ablation settings can be reproduced through `--fusion` and `--pretrain_loss`:

```bash
python main.py --mode pretraining --data carotid --fusion baseline --pretrain_loss contrastive
python main.py --mode pretraining --data carotid --fusion baseline --pretrain_loss tstcl
python main.py --mode pretraining --data carotid --fusion daa_cmau --pretrain_loss contrastive
python main.py --mode pretraining --data carotid --fusion daa_cmau --pretrain_loss tstcl
```

The final command corresponds to the default DynaCollab configuration.

## Arguments

| Argument | Choices | Default | Description |
| --- | --- | --- | --- |
| `--mode` | `pretraining`, `finetuning` | Required | Training stage. |
| `--data` | `carotid`, `BraTs19` | `carotid` | Dataset configuration. |
| `--fusion` | `baseline`, `daa`, `daa_cmau` | `daa_cmau` | Cross-modal collaboration strategy. |
| `--pretrain_loss` | `tstcl`, `contrastive` | `tstcl` | Pretraining objective. |
| `--pretrained_path` | path | None | Pretrained weights used for fine-tuning. |
| `--resume_checkpoint` | path | None | Checkpoint used to resume a training run. |
| `--tf` | `no_tf`, `all_tf` | `no_tf` | Augmentation setting. |
| `--train_ratio` | float | `0.8` | Training split ratio. |
| `--split_seed` | int | `42` | Random seed for train/validation splitting. |

## Project Structure

```text
DynaCollab/
  main.py                 # Command-line entry point
  config.py               # Experiment and path configuration
  dataset.py              # Carotid and BraTS19 dataset loaders
  trainer.py              # Pretraining and fine-tuning loops
  losses.py               # TSTCL, contrastive, Dice, and CE losses
  metrics.py              # Segmentation metrics
  models/dynacollab.py    # Dual-modal backbone with DAA/CMAU
  training/               # Runtime builders and launch utilities
```

## Sanity Checks

```bash
python -m compileall -q .
python main.py --help
python -c "from config import Config, PRETRAINING; Config(PRETRAINING, data='carotid')"
```

## Citation

If this code is useful for your research, please cite the corresponding DynaCollab paper.
