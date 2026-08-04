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


class CnnEncoder(nn.Module):
    """
    Strided CNN followed by flattening to a FFN, encoding a z-stack of images
    into a latent vector.

    [B, Cin, H, W] -> [B, D, Hout, Wout] -> [B, D*Hout*Wout] -> [B, E]
    """
    def __init__(
            self,
            in_channels: int,
            spatial_hidden_channels: Sequence[int],
            embedding_dims: int,
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

        self.embed = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(flat_dim, embedding_dims),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(self.cnn(x))


class CnnDecoder(CnnEncoder):
    """
    Extends CnnEncoder with a linear head, for decoding aberration coefficients
    from a z-stack of images.

    [B, Cin, H, W] -> [B, E] -> [B, Cout]
    """
    def __init__(
            self,
            in_channels: int,
            spatial_hidden_channels: Sequence[int],
            embedding_dims: int,
            out_dims: int,
            spatial_size: int,
    ):
        super().__init__(
            in_channels=in_channels,
            spatial_hidden_channels=spatial_hidden_channels,
            embedding_dims=embedding_dims,
            spatial_size=spatial_size,
        )
        self.project = nn.Linear(embedding_dims, out_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.embed(self.cnn(x)))


class CnnDecoder_DetachedHead(CnnDecoder):
    """
    Extends CnnDecoder with a second head from a detached copy of the embedding.

    Allows aux outputs (e.g., uncertainty params) to train without mucking up the
    main outputs in the shared trunk.

    The aux outputs are appended to the main ones along the channel dim before outputting
    """
    def __init__(
            self,
            in_channels: int,
            spatial_hidden_channels: Sequence[int],
            embedding_dims: int,
            out_dims: int,
            detached_out_dims: int,
            spatial_size: int,
    ):
        super().__init__(
            in_channels=in_channels,
            spatial_hidden_channels=spatial_hidden_channels,
            embedding_dims=embedding_dims,
            out_dims=out_dims,
            spatial_size=spatial_size,
        )
        self.detached_head = nn.Linear(embedding_dims, detached_out_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(self.cnn(x))
        main_outputs = self.project(emb)
        detached_outputs = self.detached_head(emb.detach())
        return torch.cat([main_outputs, detached_outputs], dim=1)
