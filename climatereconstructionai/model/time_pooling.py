import torch
from torch import nn


class TimePooling(nn.Module):
    # collapses the time dimension of a sequence (B, T, C, H, W) to a single
    # aggregated field (B, 1, C, H, W); "attention" learns per-step weights
    def __init__(self, mode="mean"):
        super().__init__()
        assert mode in ("none", "mean", "last", "sum", "attention")
        self.mode = mode
        if mode == "attention":
            self.score = nn.Linear(1, 1)

    def forward(self, h):
        if self.mode == "mean":
            return h.mean(dim=1, keepdim=True)
        if self.mode == "sum":
            return h.sum(dim=1, keepdim=True)
        if self.mode == "last":
            return h[:, -1:, :, :, :]
        if self.mode == "attention":
            s = h.mean(dim=(2, 3, 4))  # (B, T) per-step score
            w = torch.softmax(self.score(s.unsqueeze(-1)).squeeze(-1), dim=1)
            return (h * w.view(-1, h.shape[1], 1, 1, 1)).sum(dim=1, keepdim=True)
        return h
