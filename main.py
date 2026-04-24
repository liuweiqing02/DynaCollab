import argparse
import sys

from config import Config, FINE_TUNING, PRETRAINING
from training.runtime import run_training


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["pretraining", "finetuning"], required=True)
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Resume training from checkpoint")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Load pretrained weights for finetuning")
    parser.add_argument("--tf", type=str, choices=["no_tf", "all_tf"], default=None, help="Augmentation mode")
    parser.add_argument("--train_ratio", type=float, default=None, help="Train split ratio in (0,1)")
    parser.add_argument("--split_seed", type=int, default=None, help="Random seed used for train/val split")
    return parser.parse_args()


def main():
    args = parse_args()
    mode = PRETRAINING if args.mode == "pretraining" else FINE_TUNING
    config = Config(mode)

    if args.tf is not None:
        config.tf = args.tf
    if args.train_ratio is not None:
        config.train_ratio = float(args.train_ratio)
    if args.split_seed is not None:
        config.split_seed = int(args.split_seed)

    if args.mode == "pretraining":
        if args.resume_checkpoint:
            config.pretrained_checkpoint_path = args.resume_checkpoint
    else:
        if args.resume_checkpoint:
            config.finetuning_checkpoint_path = args.resume_checkpoint
            config.pretrained_path = None
        elif args.pretrained_path:
            config.pretrained_path = args.pretrained_path

    run_training(config)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Training failed: {e}", file=sys.stderr)
        raise
