class _PreviewModule:
    """Small callable base class so the preview stays dependency-light."""

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class DoubleConv(_PreviewModule):
    """Placeholder block name retained for architectural readability."""

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        raise NotImplementedError("Convolution details are withheld in the public preview.")


def UpConv(in_channels, out_channels):
    return {
        "module": "UpConv",
        "in_channels": in_channels,
        "out_channels": out_channels,
        "status": "withheld",
    }


class Down(_PreviewModule):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        raise NotImplementedError("Encoder details are withheld in the public preview.")


class Up(_PreviewModule):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x_down, x_up):
        raise NotImplementedError("Decoder details are withheld in the public preview.")


class CrossModalityAlignUnit(_PreviewModule):
    """High-level placeholder for cross-modality alignment."""

    def __init__(self, channels):
        self.channels = channels

    def forward(self, source, target, return_params=False):
        raise NotImplementedError(
            "CMAU internals are withheld. Public behavior: estimate alignment cues from one modality "
            "and use them to refine paired modality features."
        )


class DynamicAnatomicalAlignment(_PreviewModule):
    """High-level placeholder for anatomy-aware feature alignment."""

    def __init__(self, in_channels, num_modalities, reduction_ratio=8, alignment_mode="key", use_cross_align=True):
        self.in_channels = in_channels
        self.num_modalities = num_modalities
        self.reduction_ratio = reduction_ratio
        self.alignment_mode = alignment_mode
        self.use_cross_align = use_cross_align

    def forward(self, modality_features):
        raise NotImplementedError(
            "DAA internals are withheld. Public flow:\n"
            "1. extract modality-specific anatomical cues\n"
            "2. estimate cross-modal correspondence\n"
            "3. produce aligned feature maps for later fusion"
        )


class CrossModalUNet(_PreviewModule):
    """Public architecture sketch for DynaCollab."""

    def __init__(
        self,
        num_classes,
        in_channels=1,
        depth=5,
        num_modalities=2,
        mode="seg",
        growth_rate=32,
        alignment_mode="key",
        reduction_ratio=8,
        use_global_local_loss=True,
        baseline=False,
        use_cross_align=True,
    ):
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.depth = depth
        self.num_modalities = num_modalities
        self.mode = mode
        self.growth_rate = growth_rate
        self.alignment_mode = alignment_mode
        self.reduction_ratio = reduction_ratio
        self.use_global_local_loss = use_global_local_loss
        self.baseline = baseline
        self.use_cross_align = use_cross_align

    def describe(self):
        return [
            "encode each modality with a parallel 3D backbone",
            "apply dynamic anatomical alignment at intermediate scales",
            "exchange complementary information across modalities",
            "decode task-specific outputs for segmentation or pretraining",
        ]

    def forward(self, inputs):
        raise NotImplementedError(
            "DynaCollab forward pass is intentionally provided as pseudocode only.\n"
            "Pseudocode:\n"
            "1. Extract encoder features for each modality.\n"
            "2. At each intermediate scale, run DAA and optional CMAU.\n"
            "3. Merge aligned features back into modality-specific streams.\n"
            "4. Decode segmentation logits or project features for pretraining.\n"
            "5. Return task-dependent outputs."
        )
