import torch
import torch.nn.functional as F
from torch import nn

POOLING_MODES = ("mean", "sum", "last", "attention", "attention_sum")


class TimePooling(nn.Module):
    def __init__(self, n_channels, mode="mean", channel_modes=None):
        super().__init__()
        if channel_modes is None:
            channel_modes = [mode] * n_channels
        assert len(channel_modes) == n_channels, (
            f"channel_modes has {len(channel_modes)} entries, expected {n_channels}"
        )
        for m in channel_modes:
            assert m in POOLING_MODES, f"unknown time pooling mode: {m}"
        self.channel_modes = list(channel_modes)
        self.uses_attention = any(m.startswith("attention") for m in self.channel_modes)
        if self.uses_attention:
            self.score = nn.Conv2d(n_channels, n_channels, kernel_size=1)

    def weights(self, h):
        # per-channel weights over T, (B, T, C, H, W); attention modes score
        # every channel and pixel with a 1x1 conv: softmax over T for
        # "attention" (normalized, convex combination), softplus for
        # "attention_sum" (positive, unnormalized, so accumulated variables
        # are not constrained to a mean)
        B, T, C, H, W = h.shape
        w_norm = w_pos = None
        if self.uses_attention:
            s = self.score(h.flatten(0, 1)).view(B, T, C, H, W)
            if "attention" in self.channel_modes:
                w_norm = torch.softmax(s, dim=1)
            if "attention_sum" in self.channel_modes:
                w_pos = F.softplus(s)
        cols = []
        for c, m in enumerate(self.channel_modes):
            if m == "mean":
                cols.append(torch.full((B, T, H, W), 1.0 / T, device=h.device, dtype=h.dtype))
            elif m == "sum":
                cols.append(torch.ones(B, T, H, W, device=h.device, dtype=h.dtype))
            elif m == "last":
                col = torch.zeros(B, T, H, W, device=h.device, dtype=h.dtype)
                col[:, -1] = 1.0
                cols.append(col)
            elif m == "attention":
                cols.append(w_norm[:, :, c])
            elif m == "attention_sum":
                cols.append(w_pos[:, :, c])
        return torch.stack(cols, dim=2)

    def forward(self, h):
        # h: (B, T, C, H, W) -> (B, 1, C, H, W)
        return (h * self.weights(h)).sum(dim=1, keepdim=True)
