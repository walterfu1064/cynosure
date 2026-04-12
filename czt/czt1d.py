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
        self.freq = self._get_resampled_frequencies()

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
        Could be exposed to allow gain/loss, but I'm not sure why you'd want that here.
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
        while L < self.num_input_points + self.num_output_points + 1:
            L *= 2
        return L

    def _precompute_premultiplier(self, A: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Returns the `A^{-n} * W^{-n^2/2}` factor that pre-multipliers the input signal.
        Accumulates in log space before exponentiating for precision.
        """
        log_premult = -self.input_grid * torch.log(A) + 0.5*torch.pow(self.input_grid, 2) * torch.log(W)
        return log_premult.exp()

    def _precompute_chirp_kernel(self, W: torch.Tensor) -> torch.Tensor:
        """
        Calculates the `W^{-m^2/2}` factor that the pre-multiplied input signal is convolved with.
        Rearranges it into FFT order, pads it up to a power of 2, and returns the FFT.
        """
        m = torch.arange(-(self.num_input_points - 1), self.num_output_points, dtype=self.ftype)
        log_chirp = -0.5*torch.pow(m, 2) * torch.log(W)
        chirp = log_chirp.exp()

        vn = torch.zeros(self.conv_length, dtype=self.ctype)
        vn[:self.num_output_points] = chirp[self.num_input_points-1:]
        vn[self.conv_length - (self.num_input_points-1):] = chirp[:self.num_input_points-1]
        Vk = torch.fft.fft(vn)
        return Vk

    def _precompute_postmultiplier(self, W: torch.Tensor) -> torch.Tensor:
        """
        Returns the `W^{-m^2/2}` factor that post-multiplies the convolution output.
        Accumulates in log space before exponentiating for precision.
        """
        log_postmult = 0.5*self.output_grid * torch.log(W)
        return log_postmult.exp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.num_input_points, f"Input shape {x.shape} does not match num_input_points {self.num_input_points}"

        a = torch.zeros(self.conv_length, dtype=self.ctype, device=x.device)
        a[:self.num_input_points] = x * self.premultiplier
        conv = torch.fft.ifft(torch.fft.fft(a) * self.chirp_kernel)
        czt = conv[:self.num_output_points] * self.postmultiplier
        return czt
