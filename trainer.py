class Trainer:
    """Public preview trainer that exposes the experiment flow only."""

    def __init__(self, model, loss, loader_train, loader_val, config, dataset_val=None):
        self.model = model
        self.loss = loss
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.config = config
        self.dataset_val = dataset_val

    def pretraining(self):
        return [
            "build dual-modal batches",
            "encode each modality with shared design principles",
            "apply dynamic anatomical alignment between modalities",
            "optimize global or global-local pretraining objective",
            "track validation trends and save checkpoints internally",
        ]

    def fine_tuning(self):
        return [
            "load pretrained or resumed model state",
            "decode modality-aware segmentation predictions",
            "optimize segmentation loss for each modality",
            "evaluate overlap-based metrics on the validation split",
            "select the best checkpoint according to validation performance",
        ]
