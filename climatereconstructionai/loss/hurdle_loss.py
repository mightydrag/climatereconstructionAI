import torch
import torch.nn.functional as F
from torch import nn

from .. import config as cfg
from ..model.heads import hurdle_output_layout
from ..utils.normalizer import is_precip_data_type


def hurdle_channel_map():
    # maps each hurdle target channel to its 3 output channels
    # (occurrence logit, gamma shape, conditional mean)
    target_types = cfg.data_types[-cfg.out_channels:]
    hurdle_types = getattr(cfg, "hurdle_data_types", None) or []
    return {
        c: start
        for c, is_hurdle, start, n_out in hurdle_output_layout(target_types, hurdle_types)
        if is_hurdle
    }, target_types


def denormalize_target(gt_ch, target_idx, target_type, train_stats):
    # the dataloader applies log1p (precip types) + z-normalization; the
    # hurdle-gamma distribution is defined on the raw variable
    if train_stats is not None:
        idx = len(cfg.data_types) - cfg.out_channels + target_idx
        gt_ch = gt_ch * train_stats["std"][idx] + train_stats["mean"][idx]
        if is_precip_data_type(target_type):
            gt_ch = torch.expm1(gt_ch)
    return gt_ch


class HurdleGammaLoss(nn.Module):
    # hurdle-gamma NLL on the hole region: Bernoulli NLL of the occurrence
    # logit over all pixels + gamma NLL of the amount on wet pixels;
    # output channels of each hurdle variable are (logit_p, k, mu)
    def __init__(self, threshold=0.1, channels=None, train_stats=None):
        super().__init__()
        self.threshold = threshold
        self.thresholds = list(threshold) if isinstance(threshold, (list, tuple)) else None
        self.channels = channels
        self.train_stats = train_stats

    def forward(self, data_dict):
        loss_dict = {"hurdle_gamma": 0.0}
        output, gt = data_dict["output"], data_dict["gt"]
        mask, loss_mask = data_dict["mask"], data_dict["loss_mask"]

        channel_map, target_types = hurdle_channel_map()
        channel_ids = self.channels if self.channels is not None else sorted(channel_map)

        for channel in channel_ids:
            start = channel_map[channel]
            thr = self.thresholds[channel] if self.thresholds is not None else self.threshold
            logit_p = torch.unsqueeze(output[:, start, :, :], dim=1)
            k = torch.unsqueeze(output[:, start + 1, :, :], dim=1)
            mu = torch.unsqueeze(output[:, start + 2, :, :], dim=1)
            gt_ch = torch.unsqueeze(gt[:, channel, :, :], dim=1)
            gt_ch = denormalize_target(gt_ch, channel, target_types[channel], self.train_stats)
            mask_ch = torch.unsqueeze(mask[:, channel, :, :], dim=1)
            wet = (gt_ch > thr).float()

            if loss_mask is not None:
                loss_mask_ch = torch.unsqueeze(loss_mask[:, channel, :, :], dim=1)
                hole = (1 - mask_ch) * loss_mask_ch
            else:
                hole = 1 - mask_ch

            bce = F.binary_cross_entropy_with_logits(logit_p, wet, reduction="none")
            occ = (hole * bce).sum() / hole.sum().clamp(min=1.0)

            gamma = torch.distributions.Gamma(k, k / mu.clamp_min(1e-6))
            nll = -gamma.log_prob(gt_ch.clamp_min(1e-6))
            weight = hole * wet
            amt = (weight * nll).sum() / weight.sum().clamp(min=1.0)

            loss_dict["hurdle_gamma"] += occ + amt

        return loss_dict


class HurdleGammaLossValid(nn.Module):
    # same as HurdleGammaLoss but on the valid (observed) region
    def __init__(self, threshold=0.1, channels=None, train_stats=None):
        super().__init__()
        self.threshold = threshold
        self.thresholds = list(threshold) if isinstance(threshold, (list, tuple)) else None
        self.channels = channels
        self.train_stats = train_stats

    def forward(self, data_dict):
        loss_dict = {"hurdle_gamma_valid": 0.0}
        output, gt = data_dict["output"], data_dict["gt"]
        mask, loss_mask = data_dict["mask"], data_dict["loss_mask"]

        channel_map, target_types = hurdle_channel_map()
        channel_ids = self.channels if self.channels is not None else sorted(channel_map)

        for channel in channel_ids:
            start = channel_map[channel]
            thr = self.thresholds[channel] if self.thresholds is not None else self.threshold
            logit_p = torch.unsqueeze(output[:, start, :, :], dim=1)
            k = torch.unsqueeze(output[:, start + 1, :, :], dim=1)
            mu = torch.unsqueeze(output[:, start + 2, :, :], dim=1)
            gt_ch = torch.unsqueeze(gt[:, channel, :, :], dim=1)
            gt_ch = denormalize_target(gt_ch, channel, target_types[channel], self.train_stats)
            mask_ch = torch.unsqueeze(mask[:, channel, :, :], dim=1)
            wet = (gt_ch > thr).float()

            if loss_mask is not None:
                loss_mask_ch = torch.unsqueeze(loss_mask[:, channel, :, :], dim=1)
                valid = mask_ch * loss_mask_ch
            else:
                valid = mask_ch

            bce = F.binary_cross_entropy_with_logits(logit_p, wet, reduction="none")
            occ = (valid * bce).sum() / valid.sum().clamp(min=1.0)

            gamma = torch.distributions.Gamma(k, k / mu.clamp_min(1e-6))
            nll = -gamma.log_prob(gt_ch.clamp_min(1e-6))
            weight = valid * wet
            amt = (weight * nll).sum() / weight.sum().clamp(min=1.0)

            loss_dict["hurdle_gamma_valid"] += occ + amt

        return loss_dict
