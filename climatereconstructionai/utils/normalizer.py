import numpy as np
from torchvision import transforms
from .. import config as cfg


def is_precip_data_type(data_type):
    return getattr(cfg, "precip_log_transform", False) and data_type in getattr(
        cfg, "precip_log_names", []
    )


def precip_log_transform(img_data, data_type):
    if is_precip_data_type(data_type):
        return np.log1p(img_data)
    return img_data


def precip_log_inverse(img_data, data_type):
    if is_precip_data_type(data_type):
        return np.expm1(img_data)
    return img_data


def img_normalization(img_data, train_stats):
    img_std, img_mean, img_znorm = [], [], []
    if cfg.normalize_data:
        for i in range(len(img_data)):
            if train_stats is None:
                img_mean.append(np.nanmean(precip_log_transform(np.array(img_data[i]), cfg.data_types[i])))
                img_std.append(np.nanstd(precip_log_transform(np.array(img_data[i]), cfg.data_types[i])))
            else:
                img_mean.append(train_stats["mean"][i])
                img_std.append(train_stats["std"][i])
            img_znorm.append(transforms.Normalize(mean=[img_mean[-1]], std=[img_std[-1]]))

    return img_mean, img_std, img_znorm


def renormalize(img_data, img_mean, img_std):
    return img_std * img_data + img_mean


def bnd_normalization(img_mean, img_std):

    bounds = np.ones((cfg.out_channels, 2)) * np.inf

    if cfg.n_target_data == 0:
        mean_val, std_val = img_mean[:cfg.out_channels], img_std[:cfg.out_channels]
    else:
        mean_val, std_val = img_mean[-cfg.n_target_data:], img_std[-cfg.n_target_data:]

    if cfg.n_target_data == 0:
        target_types = cfg.data_types[: cfg.out_channels]
    else:
        target_types = cfg.data_types[-cfg.n_target_data :]

    k = 0
    for bound in (cfg.min_bounds, cfg.max_bounds):
        bounds[:, k] = bound

        for row, data_type in enumerate(target_types):
            if is_precip_data_type(data_type) and np.isfinite(bounds[row, k]):
                bounds[row, k] = np.log1p(bounds[row, k])

        if cfg.normalize_data:
            bounds[:, k] = (bounds[:, k] - mean_val) / std_val

        k += 1

    return bounds
