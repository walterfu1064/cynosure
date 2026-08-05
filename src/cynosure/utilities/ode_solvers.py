"""
ODE integrators for initial value problems of the form:
    dx/dt = velocity(x, t), x(t0) = x0
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch


VelocityFn = Callable[[torch.Tensor, float], torch.Tensor]  # (x, t) -> dx/dt


class ODESolver(ABC):
    """Base class for fixed-step explicit integrators"""

    @abstractmethod
    def step(self, velocity: VelocityFn, x: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        """Advances x(t) -> x(t + dt)"""
        ...

    def integrate(
            self,
            velocity: VelocityFn,
            x_0: torch.Tensor,
            num_steps: int,
            t_start: float = 0.0,
            t_end: float = 1.0,
    ) -> torch.Tensor:
        """Integrates velocity from t_start to t_end, starting at x_0"""
        assert num_steps >= 1, f"need at least one step, got {num_steps = }"

        dt = (t_end - t_start) / num_steps
        x = x_0
        for step_index in range(num_steps):
            x = self.step(velocity, x, t_start + step_index*dt, dt)
        return x


class EulerSolver(ODESolver):
    """
    Ye Olde Basic Euler Integrator.
    x[n+1] = x[n] + dt * v(x_n, t_n)
    """
    def step(self, velocity: VelocityFn, x: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        return x + dt * velocity(x, t)
