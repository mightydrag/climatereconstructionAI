import torch
import torch.nn.functional as F
from torch import nn


def gamma_nll(y, mean, shape, eps=1e-6):
    # negative log-likelihood of a gamma distribution parameterized by
    # its mean (mean) and shape (shape); rate = shape / mean
    mean = mean.clamp(min=eps)
    y = y.clamp(min=eps)
    shape_t = torch.as_tensor(shape, dtype=y.dtype, device=y.device)
    rate = shape_t / mean
    return rate * y - shape_t * torch.log(rate) + torch.lgamma(shape_t) - (shape_t - 1.0) * torch.log(y)


class GammaLoss(nn.Module):
    def __init__(self, threshold=0.0, shape=1.0, channels=None):
        super().__init__()
        self.threshold = threshold
        self.shape = shape
        self.channels = channels

    def forward(self, data_dict):
        loss_dict = {
            'gamma': 0.0
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
            mask_ch = torch.unsqueeze(mask[:, channel, :, :], dim=1)
            gt_ch = torch.unsqueeze(gt[:, channel, :, :], dim=1)
            output_ch = torch.unsqueeze(output[:, channel, :, :], dim=1)

            # predicted mean of the gamma distribution (strictly positive)
            mean = F.softplus(output_ch) + 1e-6
            # only pixels above the threshold contribute (e.g. wet pixels for precipitation)
            wet = (gt_ch > self.threshold).float()

            if loss_mask is not None:
                loss_mask_ch = torch.unsqueeze(loss_mask[:, channel, :, :], dim=1)
                combined_hole_mask_ch = (1 - mask_ch) * loss_mask_ch
            else:
                combined_hole_mask_ch = 1 - mask_ch

            weight = combined_hole_mask_ch * wet
            nll = gamma_nll(gt_ch, mean, self.shape)
            loss_dict['gamma'] += (weight * nll).sum() / weight.sum().clamp(min=1.0)

        return loss_dict


class GammaLossValid(nn.Module):
    def __init__(self, threshold=0.0, shape=1.0, channels=None):
        super().__init__()
        self.threshold = threshold
        self.shape = shape
        self.channels = channels

    def forward(self, data_dict):
        loss_dict = {
            'gamma_valid': 0.0
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
            mask_ch = torch.unsqueeze(mask[:, channel, :, :], dim=1)
            gt_ch = torch.unsqueeze(gt[:, channel, :, :], dim=1)
            output_ch = torch.unsqueeze(output[:, channel, :, :], dim=1)

            mean = F.softplus(output_ch) + 1e-6
            wet = (gt_ch > self.threshold).float()

            if loss_mask is not None:
                loss_mask_ch = torch.unsqueeze(loss_mask[:, channel, :, :], dim=1)
                combined_hole_mask_ch = mask_ch * loss_mask_ch
            else:
                combined_hole_mask_ch = mask_ch

            weight = combined_hole_mask_ch * wet
            nll = gamma_nll(gt_ch, mean, self.shape)
            loss_dict['gamma_valid'] += (weight * nll).sum() / weight.sum().clamp(min=1.0)

        return loss_dict
