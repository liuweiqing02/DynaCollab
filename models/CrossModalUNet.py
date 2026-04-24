import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True):
        super().__init__()
        self.batchnorm = batchnorm
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        if batchnorm:
            self.norm1 = nn.BatchNorm3d(out_channels)
            self.norm2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        if self.batchnorm:
            x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        if self.batchnorm:
            x = self.norm2(x)
        x = self.relu(x)
        return x


def UpConv(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True, down_mode="maxpool", batchnorm=True):
        super().__init__()
        self.pooling = pooling
        self.down_mode = down_mode
        if self.down_mode == "maxpool":
            self.maxpool = nn.MaxPool3d(2)
            self.doubleconv = DoubleConv(in_channels, out_channels, batchnorm=batchnorm)

    def forward(self, x):
        if self.pooling:
            x = self.maxpool(x)
        x = self.doubleconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, up_mode="transpose", batchnorm=True):
        super().__init__()
        self.up_mode = up_mode
        self.upconv = UpConv(in_channels, out_channels)
        self.doubleconv = DoubleConv(in_channels, out_channels, batchnorm=batchnorm)

    def forward(self, x_down, x_up):
        x_down = self.upconv(x_down)
        x = torch.cat((x_up, x_down), dim=1)
        x = self.doubleconv(x)
        return x


class CrossModalityAlignUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, max(4, channels // 4), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(max(4, channels // 4), channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_align = nn.Sequential(
            nn.Conv3d(channels, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(4, 3, kernel_size=3, padding=1),
        )

    def forward(self, source, target, return_params=False):
        channel_weights = self.channel_att(target)
        weighted_source = source * channel_weights

        transform_params = self.spatial_align(target)
        grid = self._generate_grid(transform_params)
        aligned = F.grid_sample(
            weighted_source,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        if return_params:
            return aligned, transform_params
        return aligned

    def _generate_grid(self, params):
        b, _, d, h, w = params.shape
        device = params.device
        z = torch.linspace(-1, 1, d, device=device)
        y = torch.linspace(-1, 1, h, device=device)
        x = torch.linspace(-1, 1, w, device=device)
        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing="ij")
        base_grid = torch.stack((grid_x, grid_y, grid_z), dim=0)
        base_grid = base_grid.unsqueeze(0).repeat(b, 1, 1, 1, 1)
        transformed_grid = base_grid + params
        return transformed_grid.permute(0, 2, 3, 4, 1)


class DynamicAnatomicalAlignment(nn.Module):
    def __init__(self, in_channels, num_modalities, reduction_ratio=8, alignment_mode="key", use_cross_align=True):
        super().__init__()
        self.num_modalities = num_modalities
        self.alignment_mode = alignment_mode
        self.use_cross_align = use_cross_align
        self.compressed_channels = max(in_channels // reduction_ratio, 4)

        self.structure_extractor = nn.Sequential(
            nn.Conv3d(in_channels, self.compressed_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, self.compressed_channels),
            nn.ReLU(inplace=True),
        )

        self.alignment_units = nn.ModuleList()
        if self.use_cross_align:
            self._create_alignment_units()

        self.feature_reconstructors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(self.compressed_channels, in_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(8, in_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_modalities)
            ]
        )

    def _create_alignment_units(self):
        if self.alignment_mode == "key":
            for _ in range(1, self.num_modalities):
                self.alignment_units.append(CrossModalityAlignUnit(self.compressed_channels))
        elif self.alignment_mode == "full":
            for _ in range(self.num_modalities * (self.num_modalities - 1) // 2):
                self.alignment_units.append(CrossModalityAlignUnit(self.compressed_channels))
        elif self.alignment_mode == "chain":
            for _ in range(self.num_modalities - 1):
                self.alignment_units.append(CrossModalityAlignUnit(self.compressed_channels))
        else:
            raise ValueError(f"Unsupported alignment_mode: {self.alignment_mode}")

    def forward(self, modality_features):
        struct_features = [self.structure_extractor(f) for f in modality_features]
        aligned_structs = [None] * self.num_modalities
        transform_params_list = []

        if self.use_cross_align:
            if self.alignment_mode == "key":
                aligned_structs[0] = struct_features[0]
                for i in range(1, self.num_modalities):
                    align_idx = i - 1
                    aligned, params = self.alignment_units[align_idx](
                        struct_features[i], struct_features[0], return_params=True
                    )
                    aligned_structs[i] = aligned
                    transform_params_list.append(params)
            elif self.alignment_mode == "full":
                aligned_structs = struct_features.copy()
                pair_idx = 0
                for i in range(self.num_modalities):
                    for j in range(i + 1, self.num_modalities):
                        align_ij = self.alignment_units[pair_idx](struct_features[i], struct_features[j])
                        align_ji = self.alignment_units[pair_idx](struct_features[j], struct_features[i])
                        aligned_structs[i] = (aligned_structs[i] + align_ji) / 2
                        aligned_structs[j] = (aligned_structs[j] + align_ij) / 2
                        pair_idx += 1
            elif self.alignment_mode == "chain":
                aligned_structs[0] = struct_features[0]
                for i in range(1, self.num_modalities):
                    aligned_structs[i] = self.alignment_units[i - 1](struct_features[i], aligned_structs[i - 1])
        else:
            aligned_structs = struct_features

        output_features = []
        for i in range(self.num_modalities):
            output_features.append(self.feature_reconstructors[i](aligned_structs[i]))

        return output_features, transform_params_list


class CrossModalUNet(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels=1,
        depth=5,
        num_modalities=2,
        mode="seg",
        growth_rate=32,
        use_anatomical_alignment=True,
        alignment_mode="key",
        reduction_ratio=8,
        use_global_local_loss=True,
        baseline=False,
        use_cross_align=True,
    ):
        super().__init__()
        self.depth = depth
        self.num_modalities = num_modalities
        self.mode = mode
        self.growth_rate = growth_rate

        self.use_anatomical_alignment = use_anatomical_alignment
        self.alignment_mode = alignment_mode
        self.reduction_ratio = reduction_ratio
        self.use_global_local_loss = use_global_local_loss
        self.baseline = baseline
        self.use_cross_align = use_cross_align

        if not self.baseline and not self.use_anatomical_alignment:
            raise ValueError(
                "Legacy shared-fusion path has been removed. "
                "Please use baseline=True or use_anatomical_alignment=True."
            )

        self.init_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(in_channels, growth_rate, kernel_size=3, padding=1),
                    nn.BatchNorm3d(growth_rate),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_modalities)
            ]
        )

        self.encoders = nn.ModuleList()
        for _ in range(num_modalities):
            encoder = nn.ModuleList()
            in_ch = growth_rate
            for i in range(depth):
                out_ch = growth_rate * (2 ** i)
                encoder.append(Down(in_ch, out_ch, pooling=(i > 0)))
                in_ch = out_ch
            self.encoders.append(encoder)

        if not self.baseline:
            self.anatomical_aligners = nn.ModuleList()
            self.modal_adapters = nn.ModuleList()
            for i in range(depth):
                layer_channels = growth_rate * (2 ** i)
                self.anatomical_aligners.append(
                    DynamicAnatomicalAlignment(
                        in_channels=layer_channels,
                        num_modalities=num_modalities,
                        reduction_ratio=reduction_ratio,
                        alignment_mode=alignment_mode,
                        use_cross_align=self.use_cross_align,
                    )
                )
                adapters = nn.ModuleList()
                for _ in range(num_modalities):
                    adapters.append(
                        nn.Sequential(
                            nn.Conv3d(layer_channels, layer_channels, kernel_size=1),
                            nn.BatchNorm3d(layer_channels),
                            nn.ReLU(inplace=True),
                        )
                    )
                self.modal_adapters.append(adapters)

        self.decoders = nn.ModuleList()
        for _ in range(num_modalities):
            decoder = nn.ModuleList()
            in_ch = growth_rate * (2 ** (depth - 1))
            for _ in range(depth - 1):
                out_ch = in_ch // 2
                decoder.append(Up(in_ch, out_ch))
                in_ch = out_ch
            self.decoders.append(decoder)

        self.final_convs = nn.ModuleList([nn.Conv3d(growth_rate, num_classes, kernel_size=1) for _ in range(num_modalities)])

        if mode == "pretrain":
            self.projection_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.AdaptiveAvgPool3d(1),
                        nn.Flatten(),
                        nn.Linear(growth_rate * (2 ** (depth - 1)), 512),
                        nn.ReLU(inplace=True),
                        nn.Linear(512, 256),
                    )
                    for _ in range(num_modalities)
                ]
            )
            self.local_projection_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv3d(growth_rate, 128, kernel_size=1),
                        nn.BatchNorm3d(128),
                        nn.ReLU(inplace=True),
                    )
                    for _ in range(num_modalities)
                ]
            )

        self.weight_initializer()

    def weight_initializer(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        features = []
        for mod_idx, x in enumerate(inputs):
            features.append(self.init_convs[mod_idx](x))

        encoder_features = [[] for _ in range(self.depth + 1)]
        encoder_features[0] = features

        fused_features = [None] * (self.depth + 1)
        fused_features[0] = features

        all_deform_params = []
        for layer_idx in range(1, self.depth + 1):
            current_features = []
            for mod_idx in range(self.num_modalities):
                x = encoder_features[0][mod_idx] if layer_idx == 1 else fused_features[layer_idx - 1][mod_idx]
                x = self.encoders[mod_idx][layer_idx - 1](x)
                current_features.append(x)
            encoder_features[layer_idx] = current_features

            if layer_idx < self.depth:
                if self.baseline:
                    next_features = current_features
                else:
                    aligned_features, params_this_layer = self.anatomical_aligners[layer_idx - 1](current_features)
                    all_deform_params.extend(params_this_layer)
                    next_features = []
                    for mod_idx in range(self.num_modalities):
                        adapted = self.modal_adapters[layer_idx - 1][mod_idx](aligned_features[mod_idx])
                        next_features.append(current_features[mod_idx] + adapted)
                fused_features[layer_idx] = next_features

        if self.mode == "pretrain" and not self.use_global_local_loss:
            return [self.projection_heads[mod_idx](encoder_features[self.depth][mod_idx]) for mod_idx in range(self.num_modalities)]

        decoder_features = [encoder_features[self.depth][mod_idx] for mod_idx in range(self.num_modalities)]
        skip_features = encoder_features[1:self.depth][::-1]

        for layer_idx in range(self.depth - 1):
            next_features = []
            for mod_idx in range(self.num_modalities):
                x = decoder_features[mod_idx]
                skip = skip_features[layer_idx][mod_idx]
                x = self.decoders[mod_idx][layer_idx](x, skip)
                next_features.append(x)
            decoder_features = next_features

        if self.mode == "pretrain":
            global_projections = [
                self.projection_heads[mod_idx](encoder_features[self.depth][mod_idx]) for mod_idx in range(self.num_modalities)
            ]
            local_projections = [
                self.local_projection_heads[mod_idx](decoder_features[mod_idx]) for mod_idx in range(self.num_modalities)
            ]
            return global_projections, local_projections, all_deform_params

        outputs = [self.final_convs[mod_idx](decoder_features[mod_idx]) for mod_idx in range(self.num_modalities)]
        return outputs
