# DynaCollab

This repository provides the public training code for DynaCollab, a dual-modal 3D segmentation framework with optional Dynamic Anatomical Alignment (DAA), Cross-Modality Alignment Units (CMAU), and Two-Stage Task-Aware Collaborative Contrastive Loss (TSTCL).

## Release Scope

The public release focuses on the model family used in the paper:

| Option | Description |
| --- | --- |
| `baseline` | Dual-modal backbone without DAA or CMAU. |
| `daa` | Backbone with DAA enabled. |
| `daa_cmau` | Backbone with DAA and CMAU enabled. This is the default DynaCollab model. |

Pretraining supports two objectives:

| Option | Description |
| --- | --- |
| `tstcl` | Two-Stage Task-Aware Collaborative Contrastive Loss. This is the default. |
| `contrastive` | Standard dual-modal contrastive loss. |

Standalone alternative architecture branches and local-only inference scripts are intentionally not included in this release.

## Installation

```bash
conda create -n dynacollab python=3.9
conda activate dynacollab
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and update the dataset paths for your machine. The real `.env` file is ignored by Git.

## Data Layout

For carotid CT/MRI experiments, set the following paths:

```bash
DYNACOLLAB_CAROTID_CT_TRAIN_IMAGES=./data/CarotidArtery_CT/imagesTr
DYNACOLLAB_CAROTID_CT_TRAIN_LABELS=./data/CarotidArtery_CT/labelsTr
DYNACOLLAB_CAROTID_MRI_TRAIN_IMAGES=./data/CarotidArtery_MRI/imagesTr
DYNACOLLAB_CAROTID_MRI_TRAIN_LABELS=./data/CarotidArtery_MRI/labelsTr
```

For BraTS19 experiments, set:

```bash
DYNACOLLAB_BRATS_TRAIN_DIR=./data/BraTS19/HGG
```

## Usage

Pretrain the default DynaCollab configuration:

```bash
python main.py --mode pretraining --data carotid
```

Fine-tune from a pretrained checkpoint:

```bash
python main.py --mode finetuning --data carotid --pretrained_path ./runs/<run>/checkpoints/best.pth
```

Select model and loss variants:

```bash
python main.py --mode pretraining --data carotid --fusion baseline --pretrain_loss contrastive
python main.py --mode pretraining --data carotid --fusion baseline --pretrain_loss tstcl
python main.py --mode pretraining --data carotid --fusion daa_cmau --pretrain_loss contrastive
python main.py --mode pretraining --data carotid --fusion daa_cmau --pretrain_loss tstcl
```

The last command is the default DynaCollab setting reported as the main method.

## Main Arguments

| Argument | Choices | Default |
| --- | --- | --- |
| `--mode` | `pretraining`, `finetuning` | Required |
| `--data` | `carotid`, `BraTs19` | `carotid` |
| `--fusion` | `baseline`, `daa`, `daa_cmau` | `daa_cmau` |
| `--pretrain_loss` | `tstcl`, `contrastive` | `tstcl` |
| `--pretrained_path` | path | None |
| `--resume_checkpoint` | path | None |

## Reproducibility Checks

Before training, the release can be checked with:

```bash
python -m compileall -q .
python main.py --help
python -c "from config import Config, PRETRAINING; Config(PRETRAINING, data='carotid')"
```

Training outputs are written under `runs/`. Checkpoints, datasets, local environment files, and evaluation outputs are ignored by Git.
