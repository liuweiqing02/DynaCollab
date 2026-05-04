# DynaCollab

Public preview of the **DynaCollab** framework for dual-modal 3D medical image segmentation.

This repository is intentionally simplified because the corresponding paper is still under review. The public version keeps the project structure and high-level method flow, while removing implementation details that would enable full reproduction before publication.

## What Is Included

- A lightweight project layout for the main training stages.
- Pseudocode-style model, training, loss, and evaluation modules.
- Configuration entry points that document the main experiment variants.

## What Is Not Included

- Full data preprocessing and augmentation code.
- Complete training pipeline and engineering details.
- Exact optimization, alignment, and loss implementation used in the paper.
- Private dataset organization and reproduction scripts.

## Method Sketch

```text
Input dual-modal volumes
  -> modality-specific feature extraction
  -> anatomy-aware cross-modal alignment
  -> collaborative feature exchange
  -> task-specific decoding
  -> segmentation outputs or pretraining projections
```

## Repository Layout

```text
DynaCollab/
  main.py               # CLI entry point for the public preview
  config.py             # High-level experiment configuration
  dataset.py            # Dataset interface placeholders
  trainer.py            # Training-stage pseudocode
  losses.py             # Loss placeholders and method notes
  metrics.py            # Evaluation placeholders
  models/
    dynacollab.py       # DynaCollab architecture sketch
  training/
    builders.py         # Factory helpers for preview components
    runtime.py          # Public preview launcher
```

## Usage

The current repository is intended for reading the method structure rather than reproducing the full experiments.

```bash
python main.py --mode pretraining --data carotid
python main.py --mode finetuning --data BraTs19
```

These commands print the public preview flow and explain which components have been withheld.

## Release Note

The full implementation, preprocessing details, and reproducibility assets can be released after the paper is accepted.

## Citation

If you use this repository as a reference, please cite the DynaCollab paper once the bibliographic information is available.
