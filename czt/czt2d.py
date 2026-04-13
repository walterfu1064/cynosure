"""
2D version of the Chirp Z Transform, implemented as a thin wrapper around two ChirpZTransform1D objects.

Arguments should be given in the same order as the last two tensor axes.  I.e., for a tensor with shape [B, Y, X],
each of the constructor args should be passed as (param_Y, param_X). Note that the param order differs from
the execution order, as in this example, the CZT will be calculated first along X, then Y.
"""

import torch
import torch.nn as nn

from czt.czt1d import ChirpZTransform1D


class ChirpZTransform2D(nn.Module):
    def __init__(
            self,
            num_input_points: tuple[int, int],
            num_output_points: tuple[int, int],
            input_step: tuple[float, float],
            start_frequency: tuple[float, float],
            end_frequency: tuple[float, float],
            *,
            ftype: torch.dtype = torch.float64,
            ctype: torch.dtype = torch.complex128
    ):
        super().__init__()

        self.czt_last = ChirpZTransform1D(
            num_input_points=num_input_points[1],
            num_output_points=num_output_points[1],
            input_step=input_step[1],
            start_frequency=start_frequency[1],
            end_frequency=end_frequency[1],
            ftype=ftype,
            ctype=ctype
        )

        self.czt_penult = ChirpZTransform1D(
            num_input_points=num_input_points[0],
            num_output_points=num_output_points[0],
            input_step=input_step[0],
            start_frequency=start_frequency[0],
            end_frequency=end_frequency[0],
            ftype=ftype,
            ctype=ctype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.czt_last(x)
        x = self.czt_penult(x.swapaxes(-1, -2).contiguous())
        return x.swapaxes(-1, -2).contiguous()
