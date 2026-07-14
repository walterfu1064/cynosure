import math
from collections.abc import Sequence
from itertools import pairwise
from typing import Literal, Optional

import torch
import torch.nn as nn


def conv_norm_act(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: Optional[int] = None,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        stride: int = 1,
):
    if padding is None:
        padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, padding_mode=padding_mode, stride=stride),
        nn.GroupNorm(min(8, out_channels), out_channels),
        nn.GELU(),
    )


class ZstackDecoder(nn.Module):
    """
    Strided CNN followed by flattening to a FFN, for decoding aberration
    coefficients from a z-stack of images.

    [B, Cin, H, W] -> [B, D, Hout, Wout] -> [B, E*Hout*Wout] -> [B, Cout]
    """
    def __init__(
            self,
            in_channels: int,
            spatial_hidden_channels: Sequence[int],
            embedding_dims: int,
            out_dims: int,
            spatial_size: int,
    ):
        super().__init__()

        assert len(spatial_hidden_channels) >= 1

        cnn = []
        for in_ch, out_ch in pairwise([in_channels, *spatial_hidden_channels]):
            cnn.append(conv_norm_act(in_ch, out_ch, 3))
            cnn.append(conv_norm_act(out_ch, out_ch, 3, stride=2))
        self.cnn = nn.Sequential(*cnn)

        num_downsamplings = len(spatial_hidden_channels)
        downsampled_size = int(math.ceil(spatial_size / (2 ** num_downsamplings)))
        flat_dim = spatial_hidden_channels[-1] * downsampled_size**2

        self.ffn = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(flat_dim, embedding_dims),
            nn.GELU(),
            nn.Linear(embedding_dims, out_dims),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.cnn(x))
