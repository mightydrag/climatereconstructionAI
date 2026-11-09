import torch
from torch import nn


class BernoulliLoss(nn.Module):
    def __init__(self, pos_weight=1.0, threshold=0.0):
        super().__init__()
        self.bce = nn.BCELoss(reduction="none")
        self.pos_weight = pos_weight
        self.threshold = threshold

    def forward(self, data_dict):
        loss_dict = {
            'bern': 0.0
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

            target = (gt_ch > self.threshold).float()
            pred = torch.sigmoid(output_ch).clamp(1e-6, 1 - 1e-6)
            weight = 1.0 + (self.pos_weight - 1.0) * target

            if loss_mask is not None:
                loss_mask_ch = torch.unsqueeze(loss_mask[:, channel, :, :], dim=1)
                combined_hole_mask_ch = (1 - mask_ch) * loss_mask_ch
            else:
                combined_hole_mask_ch = 1 - mask_ch

            loss_dict['bern'] += (weight * combined_hole_mask_ch * self.bce(pred, target)).mean()

        return loss_dict