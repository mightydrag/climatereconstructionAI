import json
import os
import random

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, Sampler

from .netcdfchecker import dataset_formatter
from .normalizer import img_normalization, bnd_normalization, is_precip_data_type, precip_log_transform
from .. import config as cfg


def load_steadymask(path, mask_names, data_types, device):
    if mask_names is None:
        return None
    else:
        assert len(mask_names) == cfg.out_channels
        if cfg.n_target_data == 0:
            steady_mask = load_netcdf(path, mask_names, data_types[: cfg.out_channels])[
                0
            ]
        else:
            steady_mask = load_netcdf(
                path, mask_names, data_types[-cfg.n_target_data :]
            )[0]
        # stack + squeeze ensures that it works with steady masks with one timestep or no timestep
        return torch.stack(
            [torch.from_numpy(np.array(mask)).to(device) for mask in steady_mask]
        ).squeeze()


class InfiniteSampler(Sampler):
    def __init__(self, num_samples, data_source=None):
        super().__init__()
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        if getattr(cfg, "target_aggregation", "none") != "none":
            # samples are already indexed in target space; windows are clamped
            n_samples = self.num_samples
            order = np.random.permutation(n_samples)
        else:
            n_samples = self.num_samples - sum(cfg.time_steps)
            order = np.random.permutation(n_samples) + cfg.time_steps[0]
        while True:
            yield order[i]
            i += 1
            if i >= n_samples:
                if getattr(cfg, "target_aggregation", "none") != "none":
                    order = np.random.permutation(n_samples)
                else:
                    order = np.random.permutation(n_samples) + cfg.time_steps[0]
                i = 0


class FiniteSampler(Sampler):
    def __init__(self, num_samples, data_source=None):
        super().__init__()
        self.num_samples = num_samples

    def __iter__(self):
        if getattr(cfg, "target_aggregation", "none") != "none":
            return iter(range(self.num_samples))
        return iter(range(cfg.time_steps[0], self.num_samples - cfg.time_steps[1]))

    def __len__(self):
        return self.num_samples


def _input_month_mask(time_coord, input_months, basename):
    try:
        months = np.asarray(time_coord.dt.month)
    except (AttributeError, TypeError, ValueError):
        try:
            decoded = xr.decode_cf(time_coord.to_dataset(name="time"))["time"]
            months = np.asarray(decoded.dt.month)
        except Exception:
            raise ValueError(
                "--input-months requires a decodable calendar 'time' coordinate "
                "in {}, but its values could not be decoded.".format(basename)
            )
    mask = np.isin(months, list(input_months))
    if not mask.any():
        raise ValueError(
            "No time steps with months {} found in {}.".format(input_months, basename)
        )
    return mask


def nc_loadchecker(filename, data_type):
    basename = filename.split("/")[-1]

    if not os.path.isfile(filename):
        print("File {} not found.".format(filename))

    try:
        ds = xr.open_dataset(filename)
    except Exception:
        try:
            ds = xr.open_dataset(filename, decode_times=False)
        except Exception:
            raise ValueError(
                "Impossible to read {}."
                "\nPlease, check that it is a netCDF file and it is not corrupted.".format(
                    filename
                )
            )

    # ds = ds.squeeze("depth")
    if "bnds" in ds.dims:
        ds = ds.drop_dims("bnds")

    ds1 = dataset_formatter(ds, data_type, basename)

    input_months = getattr(cfg, "input_months", None)
    if input_months is not None:
        if "time" not in ds1.dims:
            raise ValueError(
                "--input-months requires a 'time' dimension in {}.".format(basename)
            )
        ds1 = ds1.isel(time=_input_month_mask(ds1["time"], input_months, basename))

    ds = ds.drop_vars(data_type)

    if cfg.lazy_load:
        data = ds1[data_type]
    else:
        data = ds1[data_type].values

    dims = ds1[data_type].dims
    coords = {
        key: ds1[data_type].coords[key]
        for key in ds1[data_type].coords
        if key != "time"
    }
    ds1 = ds1.drop_vars(ds1.keys())
    ds1 = ds1.drop_dims("time")

    return [ds, ds1, dims, coords], data, data.shape[0], data.shape[1:]


def load_netcdf(path, data_names, data_types, keep_dss=False):
    if data_names is None:
        return None, None, None
    else:
        assert len(data_names) == len(data_types), (
                f"Number of data names ({len(data_names)}) does not match number of "
                f"data types ({len(data_types)}).\ndata-names: {data_names}\n"
                f"data-types: {data_types}"
            )
        if all(".nc" in data_name or ".h5" in data_name for data_name in data_names):
            data_paths = [
                [os.path.join(path, data_names[i])] for i in range(len(data_names))
            ]
        elif all(".json" in data_name for data_name in data_names):
            data_paths = []
            for data_name in data_names:
                with open(os.path.join(path, data_name)) as json_file:
                    data_paths.append(json.load(json_file))
        else:
            raise ValueError(
                'Unsupported filetype. All data names must uniformly contain ".nc", ".h5" or ".json".'
            )

        dss, data, lengths, sizes = [], [], [], []
        for i in range(len(data_paths)):
            dss_list, data_list, length, size = zip(
                *[
                    nc_loadchecker(data_paths[i][j], data_types[i])
                    for j in range(len(data_paths[i]))
                ]
            )
            dss.append(dss_list)
            data.append(data_list)
            lengths.append(length)
            sizes.append(size[0])

    agg_active = getattr(cfg, "target_aggregation", "none") != "none" and cfg.n_target_data > 0
    if agg_active and len(data_names) > cfg.n_target_data:
        # with --target-aggregation the last n_target_data entries (targets)
        # have the target (e.g. daily) time axis; input entries may be
        # sub-daily (length a multiple of the target length) or daily as well
        n_inputs = len(lengths) - cfg.n_target_data
        assert len(set(lengths[n_inputs:])) == 1, (
            "All target data files must have the same length"
        )
        target_length = sum(lengths[-1])
        for entry in lengths[:n_inputs]:
            assert sum(entry) % target_length == 0, (
                "Each input file's time axis must be a multiple of the "
                "target time axis ({})".format(target_length)
            )
    else:
        assert len(set(lengths)) == 1

    if keep_dss:
        return dss, data, lengths, sizes
    else:
        return data, lengths, sizes


class NetCDFLoader(Dataset):
    def __init__(
        self,
        data_root,
        img_names,
        mask_root,
        mask_names,
        split,
        data_types,
        time_steps,
        train_stats=None,
    ):

        super(NetCDFLoader, self).__init__()

        self.random = random.Random(cfg.loop_random_seed)

        self.data_types = data_types
        self.time_steps = time_steps

        self.n_time_steps = sum(time_steps) + 1

        mask_path = mask_root
        if split == "infill":
            data_path = os.path.join(data_root, "test/")
            self.xr_dss, self.img_data, img_all_lengths, self.img_sizes = load_netcdf(
                data_path, img_names, data_types, keep_dss=True
            )
        else:
            if split == "train":
                data_path = os.path.join(data_root, "train/")
            else:
                data_path = "{:s}/val/".format(data_root)
                if not cfg.shuffle_masks:
                    mask_path = "{:s}/val/".format(mask_root)
            self.img_data, img_all_lengths, self.img_sizes = load_netcdf(
                data_path, img_names, data_types
            )

        self.mask_data, mask_all_lengths, _ = load_netcdf(
            mask_path, mask_names, data_types
        )

        self.all_lengths = img_all_lengths
        self.img_length = img_all_lengths[0]
        if self.mask_data is None:
            self.mask_length = self.img_length
            mask_all_lengths = img_all_lengths
        else:
            self.mask_length = mask_all_lengths[0]
            if not cfg.shuffle_masks:
                print(self.mask_length, self.img_length)
                assert self.img_length == self.mask_length
        self.mask_all_lengths = mask_all_lengths

        # sub-daily inputs with an aggregated (e.g. daily) target:
        # samples are indexed in target (daily) space, inputs are loaded with
        # a sub-daily window around sample_index * time_factor
        self.target_aggregation = getattr(cfg, "target_aggregation", "none")
        self.time_factor = 1
        self.sample_lengths = [list(entry) for entry in self.all_lengths]
        self.mask_sample_lengths = [list(entry) for entry in self.mask_all_lengths]
        self.target_time_offset = getattr(cfg, "target_time_offset", 0)
        if self.target_aggregation != "none":
            assert cfg.n_target_data > 0, (
                "--target-aggregation requires --n-target-data > 0"
            )
            n_inputs = len(self.all_lengths) - cfg.n_target_data
            assert n_inputs > 0
            target_length = sum(self.all_lengths[-1])
            # per-entry resolution: targets and daily inputs are aligned with
            # the target axis (factor 1), sub-daily inputs use their own factor
            self.entry_factor = []
            self.entry_is_daily = []
            for e, entry in enumerate(self.all_lengths):
                entry_total = sum(entry)
                if e >= n_inputs or entry_total == target_length:
                    self.entry_factor.append(1)
                    self.entry_is_daily.append(True)
                else:
                    assert entry_total % target_length == 0, (
                        "The input time axis ({}) must be a multiple of the "
                        "target time axis ({})".format(entry_total, target_length)
                    )
                    self.entry_factor.append(entry_total // target_length)
                    self.entry_is_daily.append(False)
            self.time_factor = max(self.entry_factor)
            assert 0 <= self.target_time_offset < self.time_factor, (
                "--target-time-offset ({}) must be in [0, {})".format(
                    self.target_time_offset, self.time_factor
                )
            )
            for e in range(len(self.all_lengths)):
                if self.entry_is_daily[e]:
                    self.sample_lengths[e] = list(self.all_lengths[e])
                    self.mask_sample_lengths[e] = list(self.mask_all_lengths[e])
                else:
                    self.sample_lengths[e] = [
                        entry // self.entry_factor[e] for entry in self.all_lengths[e]
                    ]
                    self.mask_sample_lengths[e] = [
                        entry // self.entry_factor[e] for entry in self.mask_all_lengths[e]
                    ]
        qdm_input_file = getattr(cfg, "qdm_input_file", None)
        if qdm_input_file:
            from ..quantile_mapping import load_qdm

            qdm = load_qdm(qdm_input_file)
            if qdm.group_by_month:
                raise ValueError(
                    "qdm_input_file was fitted with monthly QDM groups, but the "
                    "loader has no time coordinate. Either fit with annual groups "
                    "or pre-apply the QDM to the files with "
                    "'python -m climatereconstructionai.quantile_mapping apply'"
                )
            for i in range(len(self.img_data)):
                if self.data_types[i] in qdm.data_types:
                    self.img_data[i] = [
                        qdm.apply(self.data_types[i], np.asarray(a))
                        for a in self.img_data[i]
                    ]
        if getattr(cfg, "precip_log_transform", False):
            for i in range(len(self.img_data)):
                if is_precip_data_type(self.data_types[i]):
                    self.img_data[i] = [
                        np.log1p(np.asarray(a)) for a in self.img_data[i]
                    ]

        self.img_mean, self.img_std, self.img_znorm = img_normalization(
            self.img_data, train_stats
        )

        self.bounds = bnd_normalization(self.img_mean, self.img_std)

    def load_data(self, ind_data, img_indices, ds_index, mask_indices, mask_ds_index):

        if self.mask_data is None:
            # Get masks from images
            image = np.array(self.img_data[ind_data][mask_ds_index][mask_indices])
            mask = torch.from_numpy((1 - (np.isnan(image))).astype(image.dtype))
        else:
            mask = torch.from_numpy(
                np.array(self.mask_data[ind_data][mask_ds_index][mask_indices])
            )
        image = np.array(self.img_data[ind_data][ds_index][img_indices])
        image = torch.from_numpy(np.nan_to_num(image))

        if cfg.normalize_data:
            image = self.img_znorm[ind_data](image)

        return image, mask

    def get_single_item(
        self, ind_data, index, shuffle_masks, mask_indices=None, mask_ds_index=None,
        mask_sample=None,
    ):
        agg = self.target_aggregation != "none"
        is_daily = agg and self.entry_is_daily[ind_data]

        # in aggregation mode samples are indexed in target (e.g. daily) space
        entry_lengths = self.all_lengths[ind_data] if agg else self.img_length
        search_lengths = self.sample_lengths[ind_data] if agg else self.img_length

        # get index of dataset
        ds_index = 0
        current_index = 0
        for f in range(len(search_lengths)):
            if index > current_index + search_lengths[f]:
                current_index += search_lengths[f]
                ds_index += 1
        index -= current_index

        if agg and is_daily:
            # the aggregated target / daily inputs are single (daily) steps,
            # repeated over the window
            center = index
            img_indices = np.full(self.n_time_steps, index)
        else:
            # sub-daily inputs use a window around the sample's center step
            # (day start shifted by --target-time-offset)
            center = index * self.entry_factor[ind_data] + self.target_time_offset if agg else index
            img_indices = np.array(
                list(range(center - self.time_steps[0], center + self.time_steps[1] + 1))
            )
        img_indices[img_indices < 0] = 0
        img_indices[img_indices > entry_lengths[ds_index] - 1] = (
            entry_lengths[ds_index] - 1
        )
        if shuffle_masks and agg:
            # mask sample drawn once in target (sample) space in __getitem__
            if mask_sample is None:
                mask_indices = img_indices
                mask_ds_index = ds_index
            else:
                mask_lengths = self.mask_all_lengths[ind_data]
                mask_search = self.mask_sample_lengths[ind_data]
                mask_ds_index = 0
                current_index = 0
                for f in range(len(mask_search)):
                    if mask_sample > current_index + mask_search[f]:
                        current_index += mask_search[f]
                        mask_ds_index += 1
                mask_index = mask_sample - current_index
                if is_daily:
                    mask_indices = np.full(self.n_time_steps, mask_index)
                else:
                    mask_center = mask_index * self.entry_factor[ind_data] + self.target_time_offset
                    mask_indices = np.array(
                        list(
                            range(
                                mask_center - self.time_steps[0],
                                mask_center + self.time_steps[1] + 1,
                            )
                        )
                    )
                mask_indices[mask_indices < 0] = 0
                mask_indices[mask_indices > mask_lengths[mask_ds_index] - 1] = (
                    mask_lengths[mask_ds_index] - 1
                )
        elif shuffle_masks and mask_indices is None:
            mask_index = self.random.randint(0, sum(self.mask_length) - 1)
            mask_ds_index = 0
            current_index = 0
            for l in range(len(self.mask_length)):
                if mask_index > current_index + self.mask_length[l]:
                    current_index += self.mask_length[l]
                    mask_ds_index += 1
            mask_index -= current_index

            # define range of lstm or prev-next steps -> adjust, if out of boundaries
            mask_indices = np.array(
                list(
                    range(
                        mask_index - self.time_steps[0],
                        mask_index + self.time_steps[1] + 1,
                    )
                )
            )
            mask_indices[mask_indices < 0] = 0
            mask_indices[mask_indices > self.mask_length[mask_ds_index] - 1] = (
                self.mask_length[mask_ds_index] - 1
            )
        elif shuffle_masks and mask_indices is not None:
            mask_indices = mask_indices
            mask_ds_index = mask_ds_index
        else:
            mask_indices = img_indices
            mask_ds_index = ds_index
        # load data from ranges
        images, masks = self.load_data(
            ind_data, img_indices, ds_index, mask_indices, mask_ds_index
        )

        # stack to correct dimensions
        images = torch.stack([images], dim=1)
        masks = torch.stack([masks], dim=1)

        return images, masks, mask_indices, mask_ds_index

    def __getitem__(self, index):

        images = []
        masks = []
        masked = []
        ndata = len(self.data_types)
        mask_indices = None
        mask_ds_index = None
        mask_sample = None
        if self.target_aggregation != "none" and cfg.shuffle_masks:
            mask_sample = self.random.randint(0, len(self) - 1)

        for i in range(ndata):

            image, mask, mask_indices, mask_ds_index = self.get_single_item(
                i, index, cfg.shuffle_masks, mask_indices, mask_ds_index, mask_sample
            )

            if i >= ndata - cfg.n_target_data:
                images.append(image)
            else:
                if cfg.n_target_data == 0:
                    images.append(image)
                masks.append(mask)
                masked.append(image * mask)

        if cfg.channel_steps:
            return (
                torch.cat(masked, dim=0).transpose(0, 1),
                torch.cat(masks, dim=0).transpose(0, 1),
                torch.cat(images, dim=0).transpose(0, 1),
                index,
            )
        else:
            return (
                torch.cat(masked, dim=1),
                torch.cat(masks, dim=1),
                torch.cat(images, dim=1),
                index,
            )

    def __len__(self):
        if self.target_aggregation != "none":
            # all entries share the same sample (target-space) index range
            return sum(self.sample_lengths[-1])
        return sum(self.img_length)
