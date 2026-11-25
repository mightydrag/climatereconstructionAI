import numpy as np
import torch

from .feature_loss import FeatureLoss
from .hole_loss import HoleLoss
from .total_variation_loss import TotalVariationLoss, TotalVariationLossValid
from .valid_loss import ValidLoss
from .bernoulli_loss import BernoulliLoss, BernoulliLossValid
from .precip_loss import PrecipLoss, PrecipLossValid
from .gamma_loss import GammaLoss, GammaLossValid
from .bern_gamma_loss import BernoulliGammaLoss, BernoulliGammaLossValid
from .hurdle_loss import HurdleGammaLoss, HurdleGammaLossValid
from .. import config as cfg
from ..utils.featurizer import VGG16FeatureExtractor
from ..utils.normalizer import is_precip_data_type


def select_channels(data_type_names):
    # map data type names to output channel indices; None selects all channels
    if not data_type_names:
        return None
    target_types = cfg.data_types[-cfg.out_channels:]
    return [c for c in range(len(target_types)) if target_types[c] in data_type_names]


def channel_thresholds(threshold, train_stats=None):
    # per-output-channel thresholds. For precipitation types with
    # --precip-log-transform + --normalize-data the CLI value is interpreted
    # in raw units and converted through log1p + z-normalization with the
    # training statistics, so e.g. 0.0 means "dry" in raw units
    target_types = cfg.data_types[-cfg.out_channels:]
    n = len(cfg.data_types)
    convert = getattr(cfg, "precip_log_transform", False) and getattr(
        cfg, "normalize_data", False
    ) and train_stats is not None
    thresholds = []
    for c in range(cfg.out_channels):
        t = threshold
        if convert and is_precip_data_type(target_types[c]):
            idx = n - cfg.out_channels + c
            t = (np.log1p(max(float(threshold), 0.0)) - train_stats["mean"][idx]) / train_stats["std"][idx]
        thresholds.append(t)
    return thresholds


def prepare_data_dict(img_mask, loss_mask, output, gt, tensor_keys):
    data_dict = dict(zip(list(tensor_keys),[None]*len(tensor_keys)))

    mask = img_mask
    #     mask = mask * loss_mask
    #     mask[mask < 0] = 0
    #     mask[mask > 1] = 1
    #     assert ((mask == 0) | (mask == 1)).all(), "Not all values in mask are zeros or ones!"
    aggregation = getattr(cfg, "target_aggregation", "none")
    if aggregation == "sum":
        # sub-daily outputs are aggregated (e.g. daily precipitation sum)
        output = output.sum(dim=1)
    elif aggregation == "mean":
        output = output.mean(dim=1)
    else:
        output = output[:, cfg.recurrent_steps, :, :, :]
    mask = mask[:, cfg.recurrent_steps, :, :, :]
    gt = gt[:, cfg.recurrent_steps, cfg.gt_channels, :, :]


    if loss_mask is not None:
        # Add batch and channels dimensions to loss_mask
        # Shape: (1, time, 1, latitude, longitude)
        loss_mask = loss_mask.unsqueeze(0).unsqueeze(2)
        loss_mask = loss_mask[:, cfg.recurrent_steps, :, :, :]
        data_dict['loss_mask'] = loss_mask
        output = output * loss_mask
        gt = gt * loss_mask
    else:
        data_dict['loss_mask'] = None

    data_dict['mask'] = mask
    data_dict['output'] = output
    data_dict['gt'] = gt
    if 'comp' in tensor_keys:
        if  loss_mask is not None:
            data_dict['comp'] = mask * loss_mask * gt + (1 - mask) * loss_mask * output
        else:
            data_dict['comp'] = mask * gt + (1 - mask) * output

    return data_dict


class loss_criterion(torch.nn.Module):
    def __init__(self, lambda_dict, train_stats=None):
        super().__init__()

        self.criterions = []
        self.tensors = ['output', 'gt', 'mask']
        style_added = False

        bern_thresholds = channel_thresholds(getattr(cfg, 'bern_threshold', 0.0), train_stats)
        gamma_thresholds = channel_thresholds(getattr(cfg, 'gamma_threshold', 0.0), train_stats)

        for loss, lambda_ in lambda_dict.items():
            if lambda_ > 0:
                if (loss == 'style' or loss == 'prc') and not style_added:
                    self.criterions.append(FeatureLoss(VGG16FeatureExtractor()).to(cfg.device))
                    self.tensors.append('comp')
                    style_added = True

                elif loss == 'valid':
                    self.criterions.append(ValidLoss().to(cfg.device))
                    self.tensors.append('valid')

                elif loss == 'hole':
                    self.criterions.append(HoleLoss().to(cfg.device))
                    self.tensors.append('hole')

                elif loss == 'tv':
                    self.criterions.append(TotalVariationLoss().to(cfg.device))
                    if 'comp' not in self.tensors:
                        self.tensors.append('comp')

                elif loss == 'tv_valid':
                    self.criterions.append(TotalVariationLossValid().to(cfg.device))

                elif loss == 'bern':
                    self.criterions.append(BernoulliLoss(getattr(cfg, 'bern_pos_weight', 1.0),
                                                         bern_thresholds).to(cfg.device))
                    self.tensors.append('bern')

                elif loss == 'bern_valid':
                    self.criterions.append(BernoulliLossValid(getattr(cfg, 'bern_pos_weight', 1.0),
                                                              bern_thresholds).to(cfg.device))

                elif loss == 'precip':
                    self.criterions.append(PrecipLoss(getattr(cfg, 'precip_loss_alpha', 1.0)).to(cfg.device))
                    self.tensors.append('precip')

                elif loss == 'precip_valid':
                    self.criterions.append(PrecipLossValid(getattr(cfg, 'precip_loss_alpha', 1.0)).to(cfg.device))

                elif loss == 'gamma':
                    gamma_channels = select_channels(getattr(cfg, 'gamma_data_types', None))
                    self.criterions.append(GammaLoss(gamma_thresholds,
                                                     getattr(cfg, 'gamma_shape', 1.0),
                                                     gamma_channels).to(cfg.device))
                    self.tensors.append('gamma')

                elif loss == 'gamma_valid':
                    gamma_channels = select_channels(getattr(cfg, 'gamma_data_types', None))
                    self.criterions.append(GammaLossValid(gamma_thresholds,
                                                          getattr(cfg, 'gamma_shape', 1.0),
                                                          gamma_channels).to(cfg.device))

                elif loss == 'bern_gamma':
                    gamma_channels = select_channels(getattr(cfg, 'gamma_data_types', None))
                    self.criterions.append(BernoulliGammaLoss(getattr(cfg, 'bern_pos_weight', 1.0),
                                                              bern_thresholds,
                                                              getattr(cfg, 'gamma_shape', 1.0),
                                                              gamma_channels).to(cfg.device))
                    self.tensors.append('bern_gamma')

                elif loss == 'bern_gamma_valid':
                    gamma_channels = select_channels(getattr(cfg, 'gamma_data_types', None))
                    self.criterions.append(BernoulliGammaLossValid(getattr(cfg, 'bern_pos_weight', 1.0),
                                                                   bern_thresholds,
                                                                   getattr(cfg, 'gamma_shape', 1.0),
                                                                   gamma_channels).to(cfg.device))

                elif loss == 'hurdle_gamma' and getattr(cfg, 'hurdle_data_types', None):
                    self.criterions.append(HurdleGammaLoss(getattr(cfg, 'hurdle_threshold', 0.1),
                                                           select_channels(cfg.hurdle_data_types),
                                                           train_stats).to(cfg.device))
                    self.tensors.append('hurdle_gamma')

                elif loss == 'hurdle_gamma_valid' and getattr(cfg, 'hurdle_data_types', None):
                    self.criterions.append(HurdleGammaLossValid(getattr(cfg, 'hurdle_threshold', 0.1),
                                                                select_channels(cfg.hurdle_data_types),
                                                                train_stats).to(cfg.device))
                    self.tensors.append('hurdle_gamma_valid')


    def forward(self, img_mask, loss_mask, output, gt):

        data_dict = prepare_data_dict(img_mask, loss_mask, output, gt, self.tensors)

        loss_dict = {}
        for criterion in self.criterions:
            loss_dict.update(criterion(data_dict))

        loss_dict["total"] = 0
        for loss, lambda_value in cfg.lambda_dict.items():
            if lambda_value > 0 and loss in loss_dict.keys():
                loss_w_lambda = loss_dict[loss] * lambda_value
                loss_dict["total"] += loss_w_lambda
                loss_dict[loss] = loss_w_lambda.item()

        return loss_dict


class LossComputation(torch.nn.Module):
    def __init__(self, lambda_dict, train_stats=None):
        super().__init__()
        if cfg.multi_gpus:
            self.criterion = torch.nn.DataParallel(loss_criterion(lambda_dict, train_stats))
        else:
            self.criterion = loss_criterion(lambda_dict, train_stats)

    def forward(self, img_mask, loss_mask, output, gt):
        loss_dict = self.criterion(img_mask, loss_mask, output ,gt)
        return loss_dict