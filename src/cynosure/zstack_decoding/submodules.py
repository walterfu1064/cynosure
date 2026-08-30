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

        self.embedding_dims = embedding_dims

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


class ZstackCnnEncoder(CnnEncoder):
    """
    CNN trunk that encodes a z-stack together, treating the z-planes as input channels.
    Just a wrapper around `CnnEncoder` that adds an unused `z` argument for API consistency
    with other encoders.

    Might delete this class later, I'm torn between reducing clutter vs. standardizing APIs.

    [B, Z, H, W] -> [B, E]
    """
    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Passed z-positions are unused, and are only included for API consistency.
        For this class, the stack geometry is already baked into the weights.
        """
        return super().forward(x)


class SetEncoder(nn.Module):
    """
    Strided, framewise CNN followed by several attention layers and cross-Z attention
    pooling, encoding an unordered, dynamically-sized z-stack of images into a latent vector.

    Each frame is catted with a Fourier embedding of its z-position, so staack geometry
    is per-call rather than being baked into the weights.

    [B, Z, H, W] + [B, Z] -> [B, E]
    """
    def __init__(
            self,
            spatial_hidden_channels: Sequence[int],
            image_embedding_dims: int,
            z_embedding_dims: int,
            num_attention_layers: int,
            num_attention_heads: int,
            attention_feedforward_dims: int,
            spatial_size: int,
            z_min_frequency: float,
            z_max_frequency: float,
    ):
        super().__init__()

        assert len(spatial_hidden_channels) >= 1

        self.embedding_dims = image_embedding_dims + z_embedding_dims

        self.framewise_cnn = CnnEncoder(
            in_channels=1,
            spatial_hidden_channels=spatial_hidden_channels,
            embedding_dims=image_embedding_dims,
            spatial_size=spatial_size,
        )

        self.z_encoder = FourierEmbed(z_embedding_dims, z_min_frequency, z_max_frequency)

        self.transformers = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=self.embedding_dims,
                nhead=num_attention_heads,
                batch_first=True,
                dim_feedforward=attention_feedforward_dims,
            )
            for _ in range(num_attention_layers)
        ])

        self.z_pool = nn.MultiheadAttention(
            embed_dim=self.embedding_dims,
            num_heads=num_attention_heads,
            batch_first=True,
        )

        self.z_query = nn.Parameter(
            torch.randn(1, 1, self.embedding_dims) / math.sqrt(self.embedding_dims)
        )  # query 1 latent from z-planes

        # Running stash of [B, num_heads, Z] attention weights each head applies to each z, as a diagnostic:
        self.last_pool_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Arguments:
        - x: batch of z-stacks, [B, Z, H, W]
        - z: batch of z-positions, [B, Z] or [Z,]
        Returns:
        - [B, E] single embedding vector across all z-planes
        """
        if z is None:
            raise ValueError("SetEncoder requires the z-positions of the stack")
        B, Z, H, W = x.shape
        if z.ndim == 1:  # interpret as [Z,] shared across batch
            z = z.unsqueeze(0).expand(B, Z)

        # normalize inputs framewise to mean 0, stdev 1
        x = x - x.mean((-2, -1), keepdim=True)
        x = x / (x.std((-2, -1), keepdim=True) + 1e-8)

        frame_emb = self.framewise_cnn(x.reshape(B*Z, 1, H, W)).reshape(B, Z, -1)  # [B, Z, H, W] -> [B, Z, Ef]
        z_emb = self.z_encoder(z)  # [B, Z] -> [B, Z, Ez]
        emb = torch.cat([z_emb, frame_emb], dim=-1)  # [B, Z, Ez+Ef]

        emb_out = self.transformers(emb)  # [B, Z, E]
        pool_output, pool_weights = self.z_pool(
            self.z_query.expand(B, -1, -1),
            emb_out,
            emb_out,
            average_attn_weights=False,
        )  # [B, 1, E], [B, num_heads, 1, Z]

        self.last_pool_weights = pool_weights.squeeze(-2).detach()  # stash [B, num_heads, Z] diagnostic
        return pool_output.squeeze(-2)  # [B, E]


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


class FourierEmbed(nn.Module):
    """
    Embeds a scalar using a geometric ladder of sinusoids.

    Used to embed the flow-matching time so the MLP can condition on it.
    Frequencies are geometrically spaced over [min_frequency, max_frequency] in cycles per unit input.
    Defaults are chosen under the assumption that 0 <= t <= 1.
    """
    def __init__(
            self,
            embedding_dims: int,
            min_frequency: float = 1.0,
            max_frequency: float = 100.0,
    ):
        super().__init__()

        assert embedding_dims % 2 == 0, f"embedding_dims must be even, got {embedding_dims}"
        assert 0 < min_frequency < max_frequency, (
            "min_frequency must be between 0 and max_frequency, "
            f"got {min_frequency = } and {max_frequency = }"
        )

        exponents = torch.linspace(0, 1, embedding_dims // 2)
        freqs = min_frequency * (max_frequency / min_frequency) ** exponents
        self.register_buffer("angular_freqs", 2 * torch.pi * freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
        - t: [...,] scalars to embed
        Returns:
        - [..., E] embeddings, sine and cosine halves concatenated
        """
        angles = t.unsqueeze(-1) * self.angular_freqs
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class VelocityField(nn.Module):
    """
    Conditional velocity field for flow matching in coefficient space.

    Encapsulates two modules:
    - time_embed: Fourier embedding for the flow time `t`
    - mlp: predicts `v(x_t, t | y)`

    Takes a conditioning vector `y` from an encoder in the owning PosteriorHead,
    so this module only handles the flow-space dynamics themselves.
    """
    def __init__(
            self,
            flow_dims: int,
            conditioning_dims: int,
            time_embedding_dims: int,
            hidden_dims: Sequence[int],
            is_residual: bool = True,
            residual_dims: Optional[int] = None,
    ):
        super().__init__()

        self.time_embed = FourierEmbed(time_embedding_dims)

        self.mlp = MLP(
            flow_dims + time_embedding_dims + conditioning_dims,
            hidden_dims,
            flow_dims,
            is_residual=is_residual,
            residual_dims=residual_dims,
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Takes the current point along the flow (x_t, t) and the conditioning vector y.
        Sinusoidally embeds the flow time t, concatenates (x_t, t_emb, y), and passes
        them through the MLP to predict the output velocity.

        In effect, outputs `v(x_t, t | y)`.

        Arguments:
        - x_t: [B, N] current point along the flow
        - t: [B,] flow time, in [0, 1]
        - y: [B, E] conditioning vector, from the encoded data
        Returns:
        - [B, N] predicted velocities, `v(x_t, t | y)`
        """
        t_emb = self.time_embed(t)
        emb = torch.cat([x_t, t_emb, y], dim=-1)
        return self.mlp(emb)
