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

    def attention_weights(self, h):
        # per-pixel weights over T for the attention channels only, as a dict
        # {channel: (B, T, H, W)}; fixed modes need no tensor (see apply)
        if not self.uses_attention:
            return {}
        B, T, C, H, W = h.shape
        s = self.score(h.flatten(0, 1)).view(B, T, C, H, W)
        w_norm = torch.softmax(s, dim=1) if "attention" in self.channel_modes else None
        w_pos = F.softplus(s) if "attention_sum" in self.channel_modes else None
        return {
            c: (w_norm if m == "attention" else w_pos)[:, :, c]
            for c, m in enumerate(self.channel_modes)
            if m.startswith("attention")
        }

    def weights(self, h):
        # full per-channel weights over T, (B, T, C, H, W) -- diagnostic/test
        # use; the model path uses apply() below, which never materializes the
        # fixed-mode weights over the full spatial grid
        B, T, C, H, W = h.shape
        w_attn = self.attention_weights(h)
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
            else:
                cols.append(w_attn[c])
        return torch.stack(cols, dim=2)

    def apply(self, h):
        # h: (B, T, C, H, W) -> (B, 1, C, H, W); fixed modes are applied as
        # mean/sum/last directly, only attention channels get a weight tensor
        B, T, C, H, W = h.shape
        out = h.new_zeros(B, 1, C, H, W)
        w_attn = self.attention_weights(h) if self.uses_attention else {}
        for c, m in enumerate(self.channel_modes):
            hc = h[:, :, c:c + 1]
            if m == "mean":
                out[:, :, c:c + 1] = hc.mean(dim=1, keepdim=True)
            elif m == "sum":
                out[:, :, c:c + 1] = hc.sum(dim=1, keepdim=True)
            elif m == "last":
                out[:, :, c:c + 1] = hc[:, -1:]
            else:
                out[:, :, c:c + 1] = (hc * w_attn[c].unsqueeze(2)).sum(dim=1, keepdim=True)
        return out

    def apply_state(self, s, n_steps, w_attn):
        # pool a recurrent state s: (B, C, H, W) over the (broadcast) time axis
        # with the same weights used for the features: mean/last leave the
        # state unchanged, sum scales it by T, attention uses w_attn
        out = s.clone()
        for c, m in enumerate(self.channel_modes):
            if c in w_attn:
                out[:, c] = (s[:, c].unsqueeze(1) * w_attn[c]).sum(dim=1)
            elif m == "sum":
                out[:, c] = s[:, c] * n_steps
        return out

    def forward(self, h):
        # h: (B, T, C, H, W) -> (B, 1, C, H, W)
        return self.apply(h)
