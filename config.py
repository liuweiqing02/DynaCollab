import json
import os
from datetime import datetime

PRETRAINING = 0
FINE_TUNING = 1

FUSION_BASELINE = "baseline"
FUSION_DAA = "daa"
FUSION_DAA_CMAU = "daa_cmau"


class Config:
    """High-level configuration kept for the public preview release."""

    def __init__(self, mode, data="carotid"):
        if mode not in {PRETRAINING, FINE_TUNING}:
            raise ValueError(f"Unknown mode: {mode}")

        self.mode = mode
        self.data = data
        self.model = "DynaCollab"
        self.fusion_strategy = FUSION_DAA_CMAU
        self.pretrain_loss = "tstcl"
        self.use_global_local_loss = True

        self.num_modalities = 2
        self.num_classes = 4 if data == "BraTs19" else 2
        self.in_channels = 2 if data == "BraTs19" else 1

        self.batch_size = 1
        self.nb_epochs = 0
        self.lr = 0.0

        self.output_root = "./runs"
        self.public_release = True
        self.release_note = (
            "This preview omits data preprocessing, optimization details, "
            "and key implementation logic until publication."
        )
        self.refresh_run_paths()

    def _build_run_name(self):
        stage = "PT" if self.mode == PRETRAINING else "FT"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{stage}_{self.data}_{self.model}_{self.fusion_strategy}_{timestamp}"

    def refresh_run_paths(self):
        self.run_name = self._build_run_name()
        self.run_dir = os.path.join(self.output_root, self.run_name)
        self.log_dir = os.path.join(self.run_dir, "logs")
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")

    @property
    def baseline(self):
        return self.fusion_strategy == FUSION_BASELINE

    @property
    def use_cross_align(self):
        return self.fusion_strategy == FUSION_DAA_CMAU

    def ensure_output_dirs(self):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def to_dict(self):
        return {
            "mode": self.mode,
            "data": self.data,
            "model": self.model,
            "fusion_strategy": self.fusion_strategy,
            "pretrain_loss": self.pretrain_loss,
            "use_global_local_loss": self.use_global_local_loss,
            "num_modalities": self.num_modalities,
            "num_classes": self.num_classes,
            "in_channels": self.in_channels,
            "public_release": self.public_release,
            "release_note": self.release_note,
            "run_name": self.run_name,
        }

    def dump_json(self, file_path):
        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
