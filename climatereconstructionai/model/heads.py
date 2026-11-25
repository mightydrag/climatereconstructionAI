import torch
import torch.nn as nn
import torch.nn.functional as F


def hurdle_output_layout(target_types, hurdle_types):
    # output channel layout when the hurdle head is active: each hurdle
    # variable occupies 3 channels (occurrence logit, gamma shape, mean),
    # all other variables keep their single channel; returns a list of
    # (target_index, is_hurdle, out_start, n_out)
    hurdle_types = hurdle_types or []
    layout = []
    out_start = 0
    for c, data_type in enumerate(target_types):
        is_hurdle = data_type in hurdle_types
        n_out = 3 if is_hurdle else 1
        layout.append((c, is_hurdle, out_start, n_out))
        out_start += n_out
    return layout


class HurdleGammaHead(nn.Module):
    # distributional head for non-negative variables such as precipitation:
    # predicts occurrence logit, gamma shape k (softplus) and conditional
    # mean mu (exp) in raw log1p space per variable
    def __init__(self, in_channels, n_hurdle, k_min=0.5):
        super().__init__()
        self.n_hurdle = n_hurdle
        self.k_min = k_min
        self.conv = nn.Conv2d(in_channels, 3 * n_hurdle, 1)

    def forward(self, h, coarse_up=None):
        # h: (B, C, H, W); coarse_up: (B, n_hurdle, H, W) interpolated coarse
        # field in raw units, used as a log-space residual on the mean and
        # (by the caller) as an input channel for the occurrence logit
        logit_p, raw_k, log_mu = self.conv(h).chunk(3, dim=1)
        if coarse_up is not None:
            log_mu = log_mu + torch.log1p(coarse_up.clamp_min(0.0))
        k = F.softplus(raw_k) + self.k_min
        mu = torch.exp(log_mu.clamp(max=20.0))
        return logit_p, k, mu


def conserve_hurdle(logit_p, k, mu, coarse_pr, max_corr=10.0):
    # rescale mu multiplicatively so the pooled expected value p * mu of the
    # predictive distribution matches the coarse field (mean conservation)
    exp_pr = torch.sigmoid(logit_p) * mu
    pooled = F.adaptive_avg_pool2d(exp_pr, coarse_pr.shape[-2:])
    corr = (coarse_pr / pooled.clamp_min(1e-6)).clamp(1.0 / max_corr, max_corr)
    corr = F.interpolate(corr, size=mu.shape[-2:], mode="bilinear", align_corners=False)
    return logit_p, k, mu * corr
