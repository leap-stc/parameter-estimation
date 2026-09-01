import torch
import torch.nn as nn
import torch.nn.functional as F

from funs.time_embedding import get_sinusoidal_table


class ConditionEmbedding(nn.Module):
    """Embeds timestep t and scalar condition c into one conditioning vector."""

    def __init__(self, time_table, cond_dim: int, out_dim: int):
        super().__init__()
        self.register_buffer("time_table", time_table)

        t_dim = time_table.shape[1]
        self.time_proj = nn.Sequential(
            nn.Linear(t_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor, cond) -> torch.Tensor:
        # t: [B] int/long; cond: [B] or [B, 1]
        t = t.long().clamp(0, self.time_table.shape[0] - 1)
        t_emb = self.time_table[t]  # [B, t_dim]
        t_emb = self.time_proj(t_emb)

        if cond is not None:
            if cond.ndim == 1:
                cond = cond.unsqueeze(-1)
            c_emb = self.cond_proj(cond.float())
            
            return t_emb + c_emb

        else:
            return t_emb
        # Fuse by addition after separate projections.
    

class ResBlock1D_conditional(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb) -> torch.Tensor:
    
        h = self.conv1(x)
        h = self.norm1(h)
        h = h + self.emb_proj(emb).unsqueeze(-1)
    
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
    
        return h + self.skip(x)


class ConditionalUNet1D(nn.Module):
    """
    A simple 1D U-Net for diffusion on line signals.

    Inputs:
      x    : [B, in_channels, L]
      t    : [B] timestep indices
      cond : [B] or [B, 1] scalar condition (e.g., sine frequency)
    Output:
      predicted noise with shape [B, in_channels, L]
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        t_steps: int = 500,
        t_dim: int = 64,
        cond_dim: int = 1,
        emb_dim: int = 128,
    ):
        super().__init__()
        time_table = get_sinusoidal_table(t_steps=t_steps, t_dim=t_dim)
        self.cond_embed = ConditionEmbedding(
            time_table=time_table,
            cond_dim=cond_dim,
            out_dim=emb_dim,
        )

        # Encoder
        self.in_conv = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = ResBlock1D_conditional(base_channels, base_channels, emb_dim)
        self.pool1 = nn.Conv1d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)

        self.down2 = ResBlock1D_conditional(base_channels * 2, base_channels * 2, emb_dim)
        self.pool2 = nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)

        # Bottleneck
        self.mid = ResBlock1D_conditional(base_channels * 4, base_channels * 4, emb_dim)

        # Decoder
        self.up2 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.dec2 = ResBlock1D_conditional(base_channels * 4, base_channels * 2, emb_dim)

        self.up1 = nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.dec1 = ResBlock1D_conditional(base_channels * 2, base_channels, emb_dim)

        self.out_conv = nn.Conv1d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        emb = self.cond_embed(t, cond)

        x0 = self.in_conv(x)
    
        x1 = self.down1(x0, emb)
        x2 = self.pool1(x1)

        x3 = self.down2(x2, emb)
        x4 = self.pool2(x3)

        xm = self.mid(x4, emb)

        y2 = self.up2(xm)
        if y2.shape[-1] != x3.shape[-1]:
            y2 = F.interpolate(y2, size=x3.shape[-1], mode="nearest")
        y2 = torch.cat([y2, x3], dim=1)
        y2 = self.dec2(y2, emb)

        y1 = self.up1(y2)
        if y1.shape[-1] != x1.shape[-1]:
            y1 = F.interpolate(y1, size=x1.shape[-1], mode="nearest")
        y1 = torch.cat([y1, x1], dim=1)
        y1 = self.dec1(y1, emb)

        return self.out_conv(y1)






