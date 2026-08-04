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


class _LinearBlock(nn.Module):
    """
    Pre-normalized linear block.

    LayerNorm so it can be used in a flow matching model, where individual samples need to be traced.
    """
    def __init__(self, in_dims: int, out_dims: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dims)
        self.linear = nn.Linear(in_dims, out_dims)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(self.norm(x)))


class _ResidualLinearBlock(nn.Module):
    """
    Pre-normalized residual linear block.

    LayerNorm so it can be used in a flow matching model, where individual samples need to be traced.
    """
    def __init__(self, num_dims: int, res_dims: int):
        super().__init__()
        self.norm = nn.LayerNorm(num_dims)
        self.linear1 = nn.Linear(num_dims, res_dims)
        self.linear2 = nn.Linear(res_dims, num_dims)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.linear1(self.norm(x))
        res = self.act(res)
        res = self.linear2(res)
        return res + x


class MLP(nn.Module):
    """
    MLP, optionally residual. If residual, hidden dimensions must all be equal.
    Used to model a vector field for flow matching.

    [B, Cin] -> [B, C1] -> [B, C2] -> ... -> [B, Cout]
    """
    def __init__(
            self,
            in_dims: int,
            hidden_dims: Sequence[int],
            out_dims: int,
            is_residual: bool,
            residual_dims: Optional[int] = None,
    ):
        super().__init__()

        assert len(hidden_dims) > 1
        if is_residual:
            assert all([h == hidden_dims[0] for h in hidden_dims])  # residual MLP requires constant-size layers
            assert residual_dims is not None

        self.input = nn.Linear(in_dims, hidden_dims[0])

        hidden = []
        for cin, cout in zip(hidden_dims[:-1], hidden_dims[1:]):
            if is_residual:
                hidden.append(_ResidualLinearBlock(cin, residual_dims))
            else:
                hidden.append(_LinearBlock(cin, cout))
        self.hidden = nn.Sequential(*hidden)

        self.output = nn.Linear(hidden_dims[-1], out_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden(self.input(x)))
