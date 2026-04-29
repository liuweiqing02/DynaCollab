import json
import os
from datetime import datetime

PRETRAINING = 0
FINE_TUNING = 1

FUSION_BASELINE = "baseline"
FUSION_DAA = "daa"
FUSION_DAA_CMAU = "daa_cmau"


def _path_from_env(name, default):
    return os.environ.get(name, default)


class Config:
    def __init__(self, mode, data="carotid"):
        assert mode in {PRETRAINING, FINE_TUNING}, f"Unknown mode: {mode}"
        self.mode = mode

        self.data = data
        self.model = "DynaCollab"
        self.growth_rate = 32

        self.fusion_strategy = FUSION_DAA_CMAU
        self.pretrain_loss = "tstcl"
        self.use_global_local_loss = True

        self.input_size = (1, 128, 128, 128)
        self.desired_spacing = (1.5, 1.5, 1.5)
        self.target_size = (128, 128, 128)
        self.target_size_crop = (128, 128, 128)
        self.enable_memory_cache = True

        if self.data == "BraTs19":
            self.dir_tr = _path_from_env("DYNACOLLAB_BRATS_TRAIN_DIR", "./data/BraTS19/HGG")
            self.num_classes = 4
            self.in_channels = 2
        elif self.data == "carotid":
            self.image_dir_mod1_tr = _path_from_env(
                "DYNACOLLAB_CAROTID_CT_TRAIN_IMAGES",
                "./data/CarotidArtery_CT/imagesTr",
            )
            self.label_dir_mod1_tr = _path_from_env(
                "DYNACOLLAB_CAROTID_CT_TRAIN_LABELS",
                "./data/CarotidArtery_CT/labelsTr",
            )
            self.image_dir_mod2_tr = _path_from_env(
                "DYNACOLLAB_CAROTID_MRI_TRAIN_IMAGES",
                "./data/CarotidArtery_MRI/imagesTr",
            )
            self.label_dir_mod2_tr = _path_from_env(
                "DYNACOLLAB_CAROTID_MRI_TRAIN_LABELS",
                "./data/CarotidArtery_MRI/labelsTr",
            )
            self.num_classes = 2
            self.in_channels = 1
        else:
            raise ValueError(f"Unsupported dataset: {self.data}")

        if self.mode == PRETRAINING:
            self.mod = "pretraining"
            self.batch_size = 1
            self.batch_size_val = 1
            self.grad_accum_steps = 2
            self.nb_epochs = 100
            self.lr = 3e-5
            self.weight_decay = 1e-4
            self.tf = "no_tf"
            self.pretrained_checkpoint_path = None
            self.use_early_stopping = True
            self.early_stopping_patience = 20
            self.early_stopping_min_delta = 1e-4
            self.early_stopping_start_epoch = 1
        else:
            self.mod = "finetuning"
            self.batch_size = 1
            self.batch_size_val = 1
            self.grad_accum_steps = 2
            self.nb_epochs = 200
            self.lr = 1e-3
            self.weight_decay = 1e-4
            self.tf = "no_tf"
            self.pretrained_path = None
            self.finetuning_checkpoint_path = None
            self.use_early_stopping = True
            self.early_stopping_patience = 25
            self.early_stopping_min_delta = 1e-4
            self.early_stopping_start_epoch = 41

        self.pin_mem = True
        self.num_cpu_workers = 8
        self.cuda = True
        self.train_ratio = 0.8
        self.split_seed = 42

        self.output_root = "./runs"
        self.checkpoint_name = "model"
        self.refresh_run_paths()

    def _build_run_name(self):
        stage = "PT" if self.mode == PRETRAINING else "FT"
        fusion_tag = {
            FUSION_BASELINE: "MV0",
            FUSION_DAA: "DAA",
            FUSION_DAA_CMAU: "DAA_CMAU",
        }[self.fusion_strategy]
        if self.mode == PRETRAINING:
            loss_tag = "TSTCL" if self.pretrain_loss == "tstcl" else "Contrastive"
        else:
            loss_tag = "SegCE_Dice"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        lr_tag = f"{self.lr:.0e}"
        accum_tag = f"acc{self.grad_accum_steps}"
        return (
            f"{stage}_{self.data}_{self.model}_{fusion_tag}_"
            f"gr{self.growth_rate}_lr{lr_tag}_bs{self.batch_size}_{accum_tag}_{loss_tag}_{timestamp}"
        )

    def refresh_run_paths(self):
        self.run_name = self._build_run_name()
        self.run_dir = os.path.join(self.output_root, self.run_name)
        self.log_dir = os.path.join(self.run_dir, "logs")
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        self.latest_checkpoint_path = os.path.join(self.checkpoint_dir, "latest.pth")
        self.best_checkpoint_path = os.path.join(self.checkpoint_dir, "best.pth")

    @property
    def baseline(self):
        return self.fusion_strategy == FUSION_BASELINE

    @property
    def use_anatomical_alignment(self):
        return not self.baseline

    @property
    def use_cross_align(self):
        return self.fusion_strategy == FUSION_DAA_CMAU

    def ensure_output_dirs(self):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def to_dict(self):
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, (str, int, float, bool, type(None), list, tuple, dict)):
                out[k] = v
            else:
                out[k] = str(v)
        out["baseline"] = self.baseline
        out["use_anatomical_alignment"] = self.use_anatomical_alignment
        out["use_cross_align"] = self.use_cross_align
        return out

    def dump_json(self, file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
