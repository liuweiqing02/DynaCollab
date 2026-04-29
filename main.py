import argparse
import sys

from config import (
    Config,
    FINE_TUNING,
    FUSION_BASELINE,
    FUSION_DAA,
    FUSION_DAA_CMAU,
    PRETRAINING,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DynaCollab variants.")
    parser.add_argument("--mode", type=str, choices=["pretraining", "finetuning"], required=True)
    parser.add_argument("--data", type=str, choices=["carotid", "BraTs19"], default="carotid")
    parser.add_argument(
        "--fusion",
        type=str,
        choices=[FUSION_BASELINE, FUSION_DAA, FUSION_DAA_CMAU],
        default=FUSION_DAA_CMAU,
        help="Model variant: baseline, DAA, or DAA/CMAU.",
    )
    parser.add_argument(
        "--pretrain_loss",
        type=str,
        choices=["tstcl", "contrastive"],
        default="tstcl",
        help="Pretraining objective.",
    )
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Resume training from checkpoint")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Load pretrained weights for finetuning")
    parser.add_argument("--tf", type=str, choices=["no_tf", "all_tf"], default=None, help="Augmentation mode")
    parser.add_argument("--train_ratio", type=float, default=None, help="Train split ratio in (0,1)")
    parser.add_argument("--split_seed", type=int, default=None, help="Random seed used for train/val split")
    parser.add_argument("--batch_size", type=int, default=None, help="Micro-batch size per optimizer forward pass")
    parser.add_argument("--batch_size_val", type=int, default=None, help="Validation batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=None, help="Gradient accumulation steps")
    return parser.parse_args()


def main():
    args = parse_args()
    mode = PRETRAINING if args.mode == "pretraining" else FINE_TUNING
    config = Config(mode, data=args.data)
    config.fusion_strategy = args.fusion
    config.pretrain_loss = args.pretrain_loss
    config.use_global_local_loss = args.pretrain_loss == "tstcl"

    if args.tf is not None:
        config.tf = args.tf
    if args.train_ratio is not None:
        config.train_ratio = float(args.train_ratio)
    if args.split_seed is not None:
        config.split_seed = int(args.split_seed)
    if args.batch_size is not None:
        config.batch_size = int(args.batch_size)
    if args.batch_size_val is not None:
        config.batch_size_val = int(args.batch_size_val)
    if args.grad_accum_steps is not None:
        config.grad_accum_steps = max(1, int(args.grad_accum_steps))

    if args.mode == "pretraining":
        if args.resume_checkpoint:
            config.pretrained_checkpoint_path = args.resume_checkpoint
    else:
        if args.resume_checkpoint:
            config.finetuning_checkpoint_path = args.resume_checkpoint
            config.pretrained_path = None
        elif args.pretrained_path:
            config.pretrained_path = args.pretrained_path

    config.refresh_run_paths()
    from training.runtime import run_training

    run_training(config)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Training failed: {e}", file=sys.stderr)
        raise
