import torch
import numpy as np


def bnd_pass(output):
    return output


def bnd_sigmoid(output, ymin, ymax):
    output = torch.sigmoid(output)
    return output * (ymax - ymin) + ymin


def bnd_exp(output, y0, sign=1):
    output = sign * torch.exp(sign * output)
    return output + y0


TEMPERATURE_TYPES = {"2t", "t2m", "tas", "tasmax", "tasmin", "t2max", "t2min"}
DEWPOINT_TYPES = {"dewpoint", "tdps", "dew_point"}


def dewpoint_temperature_pairs(target_types):
    types = [str(data_type).strip().lower() for data_type in target_types]
    temps = [i for i, t in enumerate(types) if t in TEMPERATURE_TYPES]
    dews = [i for i, t in enumerate(types) if t in DEWPOINT_TYPES]
    return [(d, t) for d in dews for t in temps]


class constrain_bounds():
    def __init__(self, bounds, target_types=None):

        self.params = []
        self.binders = []
        self.n_bounds = len(bounds)
        for i in range(self.n_bounds):
            n_inf = np.isinf(bounds[i]).sum()
            if n_inf == 2:
                self.params.append([])
                self.binders.append(bnd_pass)
            elif n_inf == 0:
                self.params.append(bounds[i])
                self.binders.append(bnd_sigmoid)
            else:
                self.binders.append(bnd_exp)
                if bounds[i, 0] == np.inf:
                    self.params.append([bounds[i, 1], -1])
                else:
                    self.params.append([bounds[i, 0]])

        if target_types is None:
            self.td_t_pairs = []
        else:
            assert len(target_types) == self.n_bounds, (
                f"Number of target types ({len(target_types)}) does not match number of "
                f"output channels ({self.n_bounds})."
            )
            self.td_t_pairs = dewpoint_temperature_pairs(target_types)

    def scale(self, output):
        channels = [
            self.binders[i](output[:, :, i, :, :], *self.params[i])
            for i in range(self.n_bounds)
        ]

        for td_idx, t_idx in self.td_t_pairs:
            channels[td_idx] = torch.minimum(channels[td_idx], channels[t_idx])

        return torch.stack(channels, dim=2)
