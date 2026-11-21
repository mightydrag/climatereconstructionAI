import torch
import torch.nn.functional as F
from torch import nn

from .gamma_loss import gamma_nll


class BernoulliGammaLoss(nn.Module):
    # two-stage loss for non-negative variables such as precipitation:
    # Bernoulli (negative log-likelihood of rain occurrence) plus a gamma
    # negative log-likelihood for the amount on wet pixels
    def __init__(self, pos_weight=1.0, threshold=0.0, shape=1.0, channels=None):
        super().__init__()
        self.bce = nn.BCELoss(reduction="none")
        self.pos_weight = pos_weight
        self.threshold = threshold
        self.thresholds = list(threshold) if isinstance(threshold, (list, tuple)) else None
        self.shape = shape
        self.channels = channels

    def forward(self, data_dict):
        loss_dict = {
            'bern_gamma': 0.0
        }

        output = data_dict['output']
        gt = data_dict['gt']
        mask = data_dict['mask']
        loss_mask = data_dict['loss_mask']

        if self.channels is None:
            channel_ids = range(output.shape[1])
        else:
            channel_ids = self.channels

        # calculate loss for the selected channels
        for channel in channel_ids:
            thr = self.thresholds[channel] if self.thresholds is not None else self.threshold
            mask_ch = torch.unsqueeze(mask[:, channel, :, :], dim=1)
            gt_ch = torch.unsqueeze(gt[:, channel, :, :], dim=1)
            output_ch = torch.unsqueeze(output[:, channel, :, :], dim=1)

            target = (gt_ch > thr).float()
            pred = torch.sigmoid(output_ch).clamp(1e-6, 1 - 1e-6)
            weight_bern = 1.0 + (self.pos_weight - 1.0) * target

            # predicted mean of the gamma distribution (strictly positive)
            mean = F.softplus(output_ch) + 1e-6
            wet = target

            if loss_mask is not None:
                loss_mask_ch = torch.unsqueeze(loss_mask[:, channel, :, :], dim=1)
                combined_hole_mask_ch = (1 - mask_ch) * loss_mask_ch
            else:
                combined_hole_mask_ch = 1 - mask_ch

            occ = (weight_bern * combined_hole_mask_ch * self.bce(pred, target)).mean()

            weight_gamma = combined_hole_mask_ch * wet
            nll = gamma_nll(gt_ch, mean, self.shape)
            amt = (weight_gamma * nll).sum() / weight_gamma.sum().clamp(min=1.0)

            loss_dict['bern_gamma'] += occ + amt

        return loss_dict


class BernoulliGammaLossValid(nn.Module):
    def __init__(self, pos_weight=1.0, threshold=0.0, shape=1.0, channels=None):
        super().__init__()
        self.bce = nn.BCELoss(reduction="none")
        self.pos_weight = pos_weight
        self.threshold = threshold
        self.thresholds = list(threshold) if isinstance(threshold, (list, tuple)) else None
        self.shape = shape
        self.channels = channels

    def forward(self, data_dict):
        loss_dict = {
            'bern_gamma_valid': 0.0
        }

        output = data_dict['output']
        gt = data_dict['gt']
        mask = data_dict['mask']
        loss_mask = data_dict['loss_mask']

        if self.channels is None:
            channel_ids = range(output.shape[1])
        else:
            channel_ids = self.channels

        # calculate loss for the selected channels
        for channel in channel_ids:
            thr = self.thresholds[channel] if self.thresholds is not None else self.threshold
            mask_ch = torch.unsqueeze(mask[:, channel, :, :], dim=1)
            gt_ch = torch.unsqueeze(gt[:, channel, :, :], dim=1)
            output_ch = torch.unsqueeze(output[:, channel, :, :], dim=1)

            target = (gt_ch > thr).float()
            pred = torch.sigmoid(output_ch).clamp(1e-6, 1 - 1e-6)
            weight_bern = 1.0 + (self.pos_weight - 1.0) * target

            mean = F.softplus(output_ch) + 1e-6
            wet = target

            if loss_mask is not None:
                loss_mask_ch = torch.unsqueeze(loss_mask[:, channel, :, :], dim=1)
                combined_hole_mask_ch = mask_ch * loss_mask_ch
            else:
                combined_hole_mask_ch = mask_ch

            occ = (weight_bern * combined_hole_mask_ch * self.bce(pred, target)).mean()

            weight_gamma = combined_hole_mask_ch * wet
            nll = gamma_nll(gt_ch, mean, self.shape)
            amt = (weight_gamma * nll).sum() / weight_gamma.sum().clamp(min=1.0)

            loss_dict['bern_gamma_valid'] += occ + amt

        return loss_dict
