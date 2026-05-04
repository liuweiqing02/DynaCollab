class _PublicDatasetBase:
    """Dataset placeholder kept only to document the expected interface."""

    def __init__(self, config, training=False, *args, **kwargs):
        super().__init__()
        self.config = config
        self.training = training

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise NotImplementedError(
            "Dataset loading and preprocessing are intentionally removed from the public preview."
        )

    @staticmethod
    def collate_fn(list_samples):
        raise NotImplementedError(
            "Batch collation is unavailable because the full data pipeline is not public."
        )

    @staticmethod
    def describe():
        return [
            "discover paired volumes for each modality",
            "load labels and modality-specific image channels",
            "apply internal preprocessing pipeline",
            "return tensors for dual-modal training",
        ]


class DatasetCarotid(_PublicDatasetBase):
    """Placeholder for the carotid CT/MRI setting."""


class DatasetBraTS19(_PublicDatasetBase):
    """Placeholder for the BraTS19 setting."""


def split_dataset(dataset, train_ratio=0.8, seed=42):
    """Public stub showing that train/validation splitting happens internally."""
    return "train_subset", "val_subset"
