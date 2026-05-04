import argparse

from config import (
    Config,
    FINE_TUNING,
    FUSION_BASELINE,
    FUSION_DAA,
    FUSION_DAA_CMAU,
    PRETRAINING,
)
from training.runtime import run_training


def parse_args():
    parser = argparse.ArgumentParser(description="Public preview for DynaCollab.")
    parser.add_argument("--mode", type=str, choices=["pretraining", "finetuning"], required=True)
    parser.add_argument("--data", type=str, choices=["carotid", "BraTs19"], default="carotid")
    parser.add_argument(
        "--fusion",
        type=str,
        choices=[FUSION_BASELINE, FUSION_DAA, FUSION_DAA_CMAU],
        default=FUSION_DAA_CMAU,
        help="Variant tag retained for documentation purposes.",
    )
    parser.add_argument(
        "--pretrain_loss",
        type=str,
        choices=["tstcl", "contrastive"],
        default="tstcl",
        help="Loss tag retained for documentation purposes.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mode = PRETRAINING if args.mode == "pretraining" else FINE_TUNING
    config = Config(mode, data=args.data)
    config.fusion_strategy = args.fusion
    config.pretrain_loss = args.pretrain_loss
    config.use_global_local_loss = args.pretrain_loss == "tstcl"
    config.refresh_run_paths()
    run_training(config)


if __name__ == "__main__":
    main()
