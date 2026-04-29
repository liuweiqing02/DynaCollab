# DynaCollab

Official implementation for:

**DynaCollab: Dynamic Collaborative Contrast for Multimodal Medical Segmentation**

DynaCollab addresses cross-modal feature isolation and high-level feature insensitivity in multimodal 3D medical image segmentation. The code provides the full reproducibility pipeline: preprocessing in the dataset loaders, training configuration, supervised contrastive representation learning, segmentation fine-tuning, inference, and evaluation utilities.

## 1. Main Features

- Dual-input 3D multimodal medical image segmentation.
- Cross-modal Dynamic Anatomical Alignment (DAA) with hierarchical residual fusion.
- Two-Stage Task-Aware Collaborative Contrastive Loss (TSTCL), combining anatomically consistent global contrast and dynamically anchored local contrast.
- Dice + cross-entropy segmentation fine-tuning initialized from the representation-learning encoder.
- Patient-wise train/validation split with saved validation IDs.
- Checkpointing, resume training, early stopping, and run-specific configuration snapshots.
- Gradient accumulation for `128 x 128 x 128` input volumes under limited GPU memory.

## 2. Project Structure

```text
.
  main.py                         # Unified training entry
  config.py                       # Experiment configuration and data path settings
  dataset.py                      # Dataset loading, preprocessing, pairing, augmentation
  augmentations.py                # 3D augmentation utilities
  CompatibleModel.py              # Representation-learning and fine-tuning loops
  losses.py                       # TSTCL, deformation regularization, and segmentation losses
  metrics.py                      # Dice, IoU, HD95, and BraTS regional metrics
  testdata.py                     # Inference and NIfTI export
  training/
    builders.py                   # Dataset/model/loss builders
    runtime.py                    # Runtime orchestration
  models/
    CrossModalUNet.py             # DynaCollab backbone
```

## 3. Environment

Recommended:

- Python 3.8+
- PyTorch with CUDA
- NVIDIA GPU with sufficient memory for 3D volumes

Common dependencies:

```bash
pip install torch torchvision torchaudio
pip install numpy scipy scikit-image nibabel pandas openpyxl scikit-learn tqdm matplotlib seaborn
```

## 4. Data Preparation

Set dataset paths with environment variables before running training or inference. See `.env.example` for all available variables. The code intentionally uses public placeholder paths by default and does not include local server paths.

### 4.1 Carotid Artery Dataset

Use `--data dongmai` for this dataset. The name is kept as a backward-compatible CLI key; it refers to the paired carotid CT/MRI dataset described in the paper.

Expected directories:

```text
CarotidArtery_CT/
  imagesTr/
  labelsTr/
  imagesTs/
  labelsTs/
CarotidArtery_MRI/
  imagesTr/
  labelsTr/
  imagesTs/
  labelsTs/
```

Configure paths, for example:

```bash
export DYNACOLLAB_CAROTID_CT_TRAIN_IMAGES=/path/to/CarotidArtery_CT/imagesTr
export DYNACOLLAB_CAROTID_CT_TRAIN_LABELS=/path/to/CarotidArtery_CT/labelsTr
export DYNACOLLAB_CAROTID_MRI_TRAIN_IMAGES=/path/to/CarotidArtery_MRI/imagesTr
export DYNACOLLAB_CAROTID_MRI_TRAIN_LABELS=/path/to/CarotidArtery_MRI/labelsTr
```

The CT and MRI files are paired by the numeric patient ID extracted from filenames. Each image must have a corresponding label file.

### 4.2 BraTS19 Dataset

Use `--data BraTs19` and set:

```bash
export DYNACOLLAB_BRATS_TRAIN_DIR=/path/to/BraTS19/HGG
export DYNACOLLAB_BRATS_VAL_DIR=/path/to/BraTS19/VAL
```

Each patient folder is expected to contain:

```text
<prefix>_t1.nii
<prefix>_t1ce.nii
<prefix>_t2.nii
<prefix>_flair.nii
<prefix>_seg.nii
```

The model uses two dual-channel modality groups:

- `X^(1) = [T1, T1ce]`
- `X^(2) = [T2, FLAIR]`

Therefore, `config.in_channels = 2` for BraTS19.

## 5. Preprocessing and Input Size

The default spatial protocol is implemented in `dataset.py`:

1. Resample each volume to `1.5 x 1.5 x 1.5 mm`.
2. Pad or center-crop to `128 x 128 x 128`.
3. Use `128 x 128 x 128` as the model input.

For training augmentation with `--tf all_tf`, image and label channels are transformed with the same random seed so spatial correspondence is preserved.

## 6. Training

### 6.1 Supervised Contrastive Representation Learning

```bash
python main.py --mode pretraining --data dongmai
```

With explicit augmentation and gradient accumulation:

```bash
python main.py --mode pretraining --data BraTs19 --tf all_tf --batch_size 1 --grad_accum_steps 2
```

### 6.2 Segmentation Fine-Tuning

```bash
python main.py --mode finetuning --data dongmai --pretrained_path ./runs/<pretrain_run>/checkpoints/best.pth
```

With 128-volume memory control:

```bash
python main.py --mode finetuning \
  --data BraTs19 \
  --pretrained_path ./runs/<pretrain_run>/checkpoints/best.pth \
  --batch_size 1 \
  --grad_accum_steps 2
```

The effective training batch size is:

```text
effective_batch_size = batch_size x grad_accum_steps
```

Gradient accumulation helps emulate a larger batch size, but it does not reduce the memory needed by a single forward/backward pass. If `128 x 128 x 128` still causes out-of-memory errors, reduce `batch_size` to 1 first, then increase `grad_accum_steps`.

### 6.3 Resume Training

Resume representation learning:

```bash
python main.py --mode pretraining --data dongmai --resume_checkpoint ./runs/<run_name>/checkpoints/latest.pth
```

Resume fine-tuning:

```bash
python main.py --mode finetuning --data dongmai --resume_checkpoint ./runs/<run_name>/checkpoints/latest.pth
```

## 7. Useful Runtime Options

```bash
python main.py --mode pretraining \
  --data BraTs19 \
  --tf all_tf \
  --train_ratio 0.8 \
  --split_seed 42 \
  --batch_size 1 \
  --batch_size_val 1 \
  --grad_accum_steps 2
```

| Argument | Meaning |
|---|---|
| `--mode` | `pretraining` or `finetuning` |
| `--data` | `dongmai` for the paired carotid CT/MRI dataset, or `BraTs19` |
| `--tf` | `no_tf` or `all_tf` augmentation mode |
| `--train_ratio` | patient-wise training split ratio |
| `--split_seed` | split random seed |
| `--batch_size` | training micro-batch size |
| `--batch_size_val` | validation batch size |
| `--grad_accum_steps` | number of gradient accumulation steps |
| `--resume_checkpoint` | resume checkpoint including optimizer/scheduler state |
| `--pretrained_path` | load representation-learning weights for fine-tuning |

## 8. Inference

```bash
python testdata.py --model_path ./runs/<run_name>/checkpoints/best.pth --config dongmai
python testdata.py --model_path ./runs/<run_name>/checkpoints/best.pth --config BraTs19
```

For carotid test data, optionally set:

```bash
export DYNACOLLAB_CAROTID_CT_TEST_IMAGES=/path/to/CarotidArtery_CT/imagesTs
export DYNACOLLAB_CAROTID_CT_TEST_LABELS=/path/to/CarotidArtery_CT/labelsTs
export DYNACOLLAB_CAROTID_MRI_TEST_IMAGES=/path/to/CarotidArtery_MRI/imagesTs
export DYNACOLLAB_CAROTID_MRI_TEST_LABELS=/path/to/CarotidArtery_MRI/labelsTs
```

Predicted NIfTI files are written to:

```text
test_result/<model_name>/
```

## 9. Output Files

Each run creates:

```text
runs/<run_name>/
  logs/
    training.log
    validation_ids.csv
  checkpoints/
    latest.pth
    best.pth
  config_snapshot.json
```

`validation_ids.csv` records the held-out patient IDs for reproducibility.

## 10. Recommended Revision Experiments

For the major revision, recommended starting settings are:

```text
input size:             128 x 128 x 128
batch_size:             1
grad_accum_steps:       2 or 3
train_ratio:            0.8
split_seed:             42
representation aug:     all_tf
fine-tuning aug:        no_tf
```

Final hyperparameters should be reported according to the actual rerun experiments used in the revised manuscript.

## 11. Code and Data Availability Notes

The in-house carotid artery dataset cannot be publicly released because of data privacy and institutional restrictions. This repository releases the implementation, preprocessing logic, training configuration, inference script, and evaluation utilities to support reproducibility.
