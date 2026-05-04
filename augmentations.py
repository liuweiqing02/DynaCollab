class _TransformPlaceholder:
    """Identity-like transform placeholder for the public preview."""

    def __call__(self, array):
        return array


class Normalize(_TransformPlaceholder):
    pass


class Blur(_TransformPlaceholder):
    pass


class Noise(_TransformPlaceholder):
    pass


class Cutout(_TransformPlaceholder):
    pass


class Flip(_TransformPlaceholder):
    pass


class Crop(_TransformPlaceholder):
    pass


class SafeTransformer:
    """Documents that a composed augmentation pipeline exists internally."""

    def __init__(self):
        self.steps = []

    def register(self, transform, probability, target):
        self.steps.append(
            {
                "transform": transform.__class__.__name__,
                "probability": probability,
                "target": target,
            }
        )

    def __call__(self, array):
        return array
