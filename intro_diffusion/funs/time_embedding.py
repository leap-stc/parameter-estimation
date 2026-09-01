import torch
import math


def get_sinusoidal_table(t_steps: int, t_dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """
    Returns the full sinusoidal embedding table.

    Args:
        t_steps   : number of timesteps (rows), e.g. 500
        t_dim     : embedding dimension (columns, must be even), e.g. 128
        max_period: controls the minimum frequency (default 10000)

    Returns:
        table : shape [t_steps, t_dim]
    """
    assert t_dim % 2 == 0, "t_dim must be even"

    half = t_dim // 2
    t = torch.arange(t_steps, dtype=torch.float32)                          # [t_steps]
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
    )                                                                        # [half]

    args = t[:, None] * freqs[None, :]                                       # [t_steps, half]
    table = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)            # [t_steps, t_dim]

    return table
