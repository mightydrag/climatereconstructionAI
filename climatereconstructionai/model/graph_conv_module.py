import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config as cfg
from ..utils.weights import weights_init


class GraphConvBlock(nn.Module):
    # graph convolution on the regular lat/lon grid: each grid cell is a node
    # and messages are passed between the 4- or 8-nearest neighbours
    # (circular in longitude for global grids, cf. --global-padding)
    def __init__(self, channels, neighbors=8, activation=None):
        super().__init__()
        assert neighbors in (4, 8), "neighbors must be 4 or 8, got {}".format(neighbors)
        self.channels = channels
        self.neighbors = neighbors
        self.self_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.neigh_conv = nn.Conv2d(channels, channels, kernel_size=1)
        if cfg.weights:
            self.self_conv.apply(weights_init(cfg.weights))
            self.neigh_conv.apply(weights_init(cfg.weights))
        self.activation = activation if activation is not None else nn.ReLU()

    def _pad(self, x):
        if getattr(cfg, "global_padding", False):
            x = F.pad(x, (1, 1, 0, 0), mode="circular")
        else:
            x = F.pad(x, (1, 1, 0, 0), mode="constant", value=0.0)
        return F.pad(x, (0, 0, 1, 1), mode="constant", value=0.0)

    def _offsets(self):
        offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
        if self.neighbors == 4:
            offsets = [(dy, dx) for dy, dx in offsets if dy == 0 or dx == 0]
        return offsets

    def forward(self, h, mask=None):
        # h: (B, C, H, W); mask (optional): (B, 1, H, W) with 1 for valid nodes
        batch, channels, height, width = h.shape
        h_pad = self._pad(h)
        if mask is not None:
            m_pad = self._pad(mask)
        else:
            m_pad = torch.ones_like(h_pad)

        agg = torch.zeros_like(h)
        count = torch.zeros_like(h)
        for dy, dx in self._offsets():
            hs = h_pad[:, :, 1 + dy:1 + dy + height, 1 + dx:1 + dx + width]
            ms = m_pad[:, :, 1 + dy:1 + dy + height, 1 + dx:1 + dx + width]
            agg = agg + hs * ms
            count = count + ms

        # mean message over the valid neighbours
        message = agg / count.clamp(min=1.0)

        out = self.self_conv(h) + self.neigh_conv(message)
        if self.activation is not None:
            out = self.activation(out)
        return out
