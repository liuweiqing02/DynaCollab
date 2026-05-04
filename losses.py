class _PreviewLoss:
    """Base loss placeholder used to document the public release."""

    def __init__(self, name, summary):
        self.name = name
        self.summary = summary

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.name} is intentionally abstracted in the public preview. Summary: {self.summary}"
        )

    def describe(self):
        return {
            "name": self.name,
            "summary": self.summary,
        }


class DualModalContrastiveLoss(_PreviewLoss):
    def __init__(self):
        super().__init__(
            name="DualModalContrastiveLoss",
            summary="encourage modality-invariant paired representations during pretraining",
        )


class CombinedGlobalLocalLoss(_PreviewLoss):
    def __init__(self, **kwargs):
        super().__init__(
            name="CombinedGlobalLocalLoss",
            summary="combine global contrastive alignment with local task-aware consistency",
        )
        self.kwargs = kwargs


class CombinedLoss(_PreviewLoss):
    def __init__(self, num_classes, ce_weight=0.2, dice_weight=0.8):
        super().__init__(
            name="CombinedLoss",
            summary="combine region overlap and class-wise supervision for segmentation fine-tuning",
        )
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
