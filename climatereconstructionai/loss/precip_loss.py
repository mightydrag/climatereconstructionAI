import torch
from torch import nn


class PrecipLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.alpha = alpha

    def forward(self, data_dict):
        loss_dict = {
            'precip': 0.0
        }

        output = data_dict['output']
        gt = data_dict['gt']
        mask = data_dict['mask']
        loss_mask = data_dict['loss_mask']

        # calculate loss for all channels
        for channel in range(output.shape[1]):
            mask_ch = torch.unsqueeze(mask[:, channel, :, :], dim=1)
            gt_ch = torch.unsqueeze(gt[:, channel, :, :], dim=1)
            output_ch = torch.unsqueeze(output[:, channel, :, :], dim=1)

            weight = 1.0 + self.alpha * gt_ch.clamp(min=0)

            if loss_mask is not None:
                loss_mask_ch = torch.unsqueeze(loss_mask[:, channel, :, :], dim=1)
                combined_hole_mask_ch = (1 - mask_ch) * loss_mask_ch
            else:
                combined_hole_mask_ch = 1 - mask_ch

            loss_dict['precip'] += self.l1(weight * combined_hole_mask_ch * output_ch,
                                           weight * combined_hole_mask_ch * gt_ch)

        return loss_dict