import torch
from torch import nn


class ValidLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, data_dict):
        loss_dict = {
            'valid': 0.0
        }
        output = data_dict['output']
        gt = data_dict['gt']
        mask = data_dict['mask']
        loss_mask = data_dict['loss_mask']

        # calculate loss for all channels
        for channel in range(output.shape[1]):
            # only select first channel
            mask_ch = torch.unsqueeze(mask[:, channel, :, :], dim=1)
            gt_ch = torch.unsqueeze(gt[:, channel, :, :], dim=1)
            output_ch = torch.unsqueeze(output[:, channel, :, :], dim=1)

            # define different loss functions from output and output_comp
            if loss_mask is not None:
                loss_mask_ch = torch.unsqueeze(loss_mask[:, channel, :, :], dim=1)
                combined_hole_mask_ch = mask_ch * loss_mask_ch
                loss_dict['valid'] += self.l1(combined_hole_mask_ch * output_ch, combined_hole_mask_ch * gt_ch)
            else:
                loss_dict['valid'] += self.l1(mask_ch * output_ch, mask_ch * gt_ch)
            
        return loss_dict