class _PreviewMetric:
    """Metric placeholder that exposes the public evaluation intent."""

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.records = []

    def __call__(self, prediction, target):
        return {"metric": self.name, "status": "withheld"}

    def update(self, *args, **kwargs):
        self.records.append("updated")

    def compute_stats(self):
        return {
            "dice": {
                "mean": None,
                "str": "withheld in public preview",
            }
        }


class SegmentationMetrics(_PreviewMetric):
    def __init__(self, num_classes, spacing=None):
        super().__init__("SegmentationMetrics")
        self.num_classes = num_classes
        self.spacing = spacing


class BraTSMetrics(_PreviewMetric):
    def __init__(self, spacing=None):
        super().__init__("BraTSMetrics")
        self.spacing = spacing

    def compute_stats(self):
        base = super().compute_stats()
        base["dice"]["regions"] = {
            "ET": {"str": "withheld"},
            "TC": {"str": "withheld"},
            "WT": {"str": "withheld"},
        }
        return base
