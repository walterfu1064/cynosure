"""
PyTorch module for using the Chirp Z Transform to calculate the Fourier transform of
an input signal at select frequencies that do not necessarily match the FFT grid.

Useful for calculating, e.g., just the spectral components in a particular, narrow, band.

The CZT radices and other factors are precomputed for the specific output frequency
range defined on module init.

Accepts multidimensional tensors, in which case the CZT is performed along the last dimension.

Follows the algorithm set out in:
        ```
        Rabiner, Schafer, and Rader, "The Chirp z-Transform Algorithm,"
        IEEE Transactions on Audio and Electroacoustics 17.2 pp. 86-92 (1969).
        https://ieeexplore.ieee.org/abstract/document/1162034
        ```

Key public methods:
- forward(): aliased by `__call__`, returns the CZT of the input tensor along its last dimension
- freq: accesses the resampled frequencies corresponding to the CZT output
"""

import torch
import torch.nn as nn


class ChirpZTransform1D(nn.Module):
    def __init__(
            self,
            num_input_points: int,
            num_output_points: int,
            input_step: float,
            start_frequency: float,
            end_frequency: float,
            *,
            ftype: torch.dtype = torch.float64,
            ctype: torch.dtype = torch.complex128
    ):
        super().__init__()

        self.num_input_points = num_input_points
        self.num_output_points = num_output_points
        self.dx = input_step
        self.start_frequency = start_frequency
        self.end_frequency = end_frequency
        self.ftype = ftype
        self.ctype = ctype

        self.register_buffer("input_grid", torch.arange(self.num_input_points, dtype=self.ftype))
        self.register_buffer("output_grid", torch.arange(self.num_output_points, dtype=self.ftype))
        self.df = self._get_frequency_step()
        self.register_buffer("freq", self._get_resampled_frequencies())

        A, W = self._get_czt_radices()
        self.conv_length = self._get_conv_length()
        self.register_buffer("premultiplier", self._precompute_premultiplier(A, W))
        self.register_buffer("chirp_kernel", self._precompute_chirp_kernel(W))
        self.register_buffer("postmultiplier", self._precompute_postmultiplier(W))

    def _get_frequency_step(self) -> float:
        """Returns the frequency step of the resampled CZT output"""
        return (self.end_frequency - self.start_frequency) / (self.num_output_points - 1)

    def _get_resampled_frequencies(self) -> torch.Tensor:
        """Returns the resampled frequencies of the CZT output"""
        return self.start_frequency + self.df * self.output_grid

    def _get_czt_radices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the complex numbers A and W that define the CZT's trajectory in the complex plane.

        A0 and W0 are hard-coded to 1 for the intended use case of pure frequency resampling.
        Could be exposed to allow gain/loss, but I'm not sure why you'd want that.
        """
        A0 = torch.ones(1, dtype=self.ctype)
        theta0 = torch.tensor(self.start_frequency * self.dx, dtype=self.ctype)
        A = A0 * torch.exp(2j*torch.pi*theta0)

        W0 = torch.ones(1, dtype=self.ctype)
        phi0 = torch.tensor(-self.df * self.dx, dtype=self.ctype)
        W = W0 * torch.exp(2j*torch.pi*phi0)

        return A, W

    def _get_conv_length(self) -> int:
        """
        Returns the length of the convolution kernel, paddded first to avoid wraparound,
        and then further to the next smallest power of 2.
        """
        L = 1
        while L < self.num_input_points + self.num_output_points - 1:
            L *= 2
        return L

    def _precompute_premultiplier(self, A: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Returns the `A^{-n} * W^{n^2/2}` factor that pre-multipliers the input signal.
        If we ever allow A0, W0 != 1, should compute in log-space to avoid over/underflow.
        """
        return torch.pow(A, -self.input_grid) * torch.pow(W, 0.5*self.input_grid**2)

    def _precompute_chirp_kernel(self, W: torch.Tensor) -> torch.Tensor:
        """
        Calculates the `W^{-m^2/2}` factor that the pre-multiplied input signal is convolved with.
        Rearranges it into FFT order, pads it up to a power of 2, and returns the FFT.
        """
        m = torch.arange(-(self.num_input_points - 1), self.num_output_points, dtype=self.ftype)
        chirp = torch.pow(W, -0.5*m**2)

        vn = torch.zeros(self.conv_length, dtype=self.ctype)
        vn[:self.num_output_points] = chirp[self.num_input_points-1:]
        vn[self.conv_length - (self.num_input_points-1):] = chirp[:self.num_input_points-1]
        Vk = torch.fft.fft(vn)
        return Vk

    def _precompute_postmultiplier(self, W: torch.Tensor) -> torch.Tensor:
        """
        Returns the `W^{m^2/2}` factor that post-multiplies the convolution output.
        If we ever allow A0, W0 != 1, should compute in log-space to avoid over/underflow.
        """
        return torch.pow(W, 0.5*self.output_grid**2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the CZT of the input signal along its last dimension"""
        assert x.shape[-1] == self.num_input_points, f"Input shape {x.shape} does not match num_input_points {self.num_input_points}"
        a = x * self.premultiplier
        A_fft = torch.fft.fft(a, n=self.conv_length, dim=-1)
        conv = torch.fft.ifft(A_fft * self.chirp_kernel, dim=-1)
        czt = conv[..., :self.num_output_points] * self.postmultiplier
        return czt
