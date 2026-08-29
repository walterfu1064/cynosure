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


class SetEncoder(nn.Module):
    """
    Strided, framewise CNN followed by several attention layers and cross-Z attention
    pooling, encoding an unordered, dynamically-sized z-stack of images into a latent vector.
    """
    def __init__(
            self,
            spatial_hidden_channels: Sequence[int],
            image_embedding_dims: int,
            z_embedding_dims: int,
            num_attention_layers: int,
            num_attention_heads: int,
            attention_feedforward_dim: int,
            spatial_size: int,
    ):
        super().__init__()

        assert len(spatial_hidden_channels) >= 1

        self.Ef = image_embedding_dims
        self.Ez = z_embedding_dims
        self.E = self.Ef + self.Ez

        self.framewise_cnn = CnnEncoder(
            in_channels=1,
            spatial_hidden_channels=spatial_hidden_channels,
            embedding_dims=self.Ef,
            spatial_size=spatial_size,
        )

        self.z_encoder = FourierEmbed(self.Ez)

        self.transformers = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=self.E,
                nhead=num_attention_heads,
                batch_first=True,
                dim_feedforward=attention_feedforward_dim,
            )
            for _ in range(num_attention_layers)
        ])

        self.z_pool = nn.MultiheadAttention(
            embed_dim=self.E,
            num_heads=num_attention_heads,
            batch_first=True,
        )

        self.z_query = nn.Parameter(
            torch.randn(1, 1, self.E) / math.sqrt(self.E)
        )  # query 1 latent from z-planes

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
        - x: batch of z-stacks, [B, Z, H, W]
        - z: batch of z-positions, [Z,]
        Returns:
        - pool_output: [B, E] single embedding vector across all z-planes
        - pool_weights: [B, num_heads, Z] attention weights each head applies to each z-plane
        """
        B, Z, H, W = x.shape

        frame_emb = self.framewise_cnn(x.reshape(B*Z, 1, H, W)).reshape(B, Z, -1)  # [B, Z, H, W] -> [B, Z, Ef]
        z_emb = self.z_encoder(z.unsqueeze(0))  # [Z,] -> [B=1, Z] -> [B=1, Z, Ez]
        z_emb = z_emb.expand((B, Z, -1))  # [B, Z, Ez]
        emb = torch.cat([z_emb, frame_emb], dim=-1)  # [B, Z, E]

        emb_out = self.transformers(emb)  # [B, Z, E]
        pool_output, pool_weights = self.z_pool(
            self.z_query.expand(B, -1, -1),
            emb_out,
            emb_out,
            average_attn_weights=False,
        )  # [B, 1, E], [B, num_heads, 1, Z]
        pool_output = pool_output.squeeze(-2)  # [B, E]
        pool_weights = pool_weights.squeeze(-2)  # [B, num_heads, Z]

        return pool_output, pool_weights


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

    Encapculates three modules:
    - encoder: CNN that encodes a z-stack into a conditioning vector `y`
    - time_embed: Fourier embedding for the flow time `t`
    - mlp: predicts `v(x_t, t | y)`
    """
    def __init__(
            self,
            in_channels: int,
            spatial_hidden_channels: Sequence[int],
            image_embedding_dims: int,
            spatial_size: int,
            flow_dims: int,
            time_embedding_dims: int,
            hidden_dims: Sequence[int],
            is_residual: bool = True,
            residual_dims: Optional[int] = None,
    ):
        super().__init__()

        self.encoder = CnnEncoder(
            in_channels=in_channels,
            spatial_hidden_channels=spatial_hidden_channels,
            embedding_dims=image_embedding_dims,
            spatial_size=spatial_size,
        )

        self.time_embed = FourierEmbed(time_embedding_dims)

        self.mlp = MLP(
            flow_dims + time_embedding_dims + image_embedding_dims,
            hidden_dims,
            flow_dims,
            is_residual=is_residual,
            residual_dims=residual_dims,
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encodes the images to embedding vectors for the velocity field to condition on.

        Arguments:
        - images: [B, Z, H, W] batched z-stacks (or batched single images)
        Returns:
        - [B, E], referred to elsewhere as `y`
        """
        return self.encoder(images)

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
