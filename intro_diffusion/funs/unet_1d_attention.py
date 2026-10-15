  import torch
import torch.nn as nn

from funs.time_embedding import get_sinusoidal_table
from funs.unet_1d import ResBlock1D_conditional, UNet1D


class SelfAttention1D(nn.Module):
    """
    Multi-head self-attention across sequence positions, with a residual connection.

    Convolutions only see a local window (~60 of 200 points in UNet1D), so the model cannot keep
    one frequency and phase over the whole signal. Attention lets every position see every other.
    A fixed sinusoidal position encoding is added so attention knows how far apart two positions are.
    The output projection is zero-initialized, so the block starts as the identity.
    """

    def __init__(self, channels: int, num_heads: int = 4, max_len: int = 1024):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        self.register_buffer("pos_table", get_sinusoidal_table(t_steps=max_len, t_dim=channels), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        L = x.shape[-1]
        h = self.norm(x).transpose(1, 2) + self.pos_table[:L]  # [B, L, C]
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + h.transpose(1, 2)


class MidBlockAttention(nn.Module):
    """Bottleneck: the usual conditional ResBlock followed by self-attention."""

    def __init__(self, channels: int, emb_dim: int, injection: str, num_heads: int):
        super().__init__()
        self.res = ResBlock1D_conditional(channels, channels, emb_dim, injection)
        self.attn = SelfAttention1D(channels, num_heads)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return self.attn(self.res(x, emb))


class AttentionUNet1D(UNet1D):
    """
    UNet1D with self-attention at the bottleneck (length L/4, i.e. 50 for L = 200).
    Same inputs, outputs, and options as UNet1D; the only change is the attention layer.
    Its weights are not interchangeable with UNet1D (the bottleneck parameters are renamed).
    """

    def __init__(self, num_heads: int = 4, **unet_kwargs):
        super().__init__(**unet_kwargs)
        self.mid = MidBlockAttention(
            channels=self.mid.conv1.out_channels,
            emb_dim=self.mid.emb_proj.in_features,
            injection=self.mid.injection,
            num_heads=num_heads,
        )
