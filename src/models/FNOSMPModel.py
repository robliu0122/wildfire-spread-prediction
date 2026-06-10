from typing import Any

import torch.nn as nn

from .SMPModel import SMPModel
from .fno_blocks import FNOBlock


class FNOSMPModel(SMPModel):
    """ResNet18-UNet with the spatial mixing at one encoder stage replaced by FNO block(s).

    Identical to SMPModel except that, after the encoder, the feature map at `fno_stage`
    is passed through `n_fno_blocks` FNO blocks (global spectral mixing) before the decoder.
    `fno_stage=5` is the bottleneck (512 ch, 4x4 with 128px crops); lower indices are
    higher-resolution stages (3 -> 128ch@16x16, 2 -> 64ch@32x32) where mode truncation is
    meaningful. Everything else (loss, optimizer, selection, data) matches the baseline,
    so the only difference vs the UNet baseline is conv -> FNO at this stage.
    """

    def __init__(
        self,
        encoder_name: str,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        fno_stage: int = 5,
        fno_modes_h: int = 4,
        fno_modes_w: int = 4,
        n_fno_blocks: int = 1,
        *args: Any,
        **kwargs: Any
    ):
        super().__init__(
            encoder_name=encoder_name,
            n_channels=n_channels,
            flatten_temporal_dimension=flatten_temporal_dimension,
            pos_class_weight=pos_class_weight,
            *args,
            **kwargs
        )
        self.save_hyperparameters()

        self.fno_stage = fno_stage
        stage_channels = self.model.encoder.out_channels[fno_stage]
        self.fno = nn.Sequential(*[
            FNOBlock(stage_channels, fno_modes_h, fno_modes_w)
            for _ in range(n_fno_blocks)
        ])

    def forward(self, x, doys=None):
        if self.hparams.flatten_temporal_dimension and len(x.shape) == 5:
            x = x.flatten(start_dim=1, end_dim=2)

        features = list(self.model.encoder(x))
        features[self.fno_stage] = self.fno(features[self.fno_stage])
        decoder_output = self.model.decoder(*features)
        return self.model.segmentation_head(decoder_output)
