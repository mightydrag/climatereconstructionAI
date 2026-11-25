import torch
import torch.nn as nn

from .attention_module import AttentionEncoderBlock
from .conv_configs import (
    init_enc_conv_configs,
    init_dec_conv_configs,
    init_enc_conv_configs_orig,
    init_dec_conv_configs_orig,
)
from .encoder_decoder import EncoderBlock, DecoderBlock, sequence_to_batch, batch_to_sequence
from .bounds_scaler import constrain_bounds
from .heads import HurdleGammaHead, hurdle_output_layout
from .graph_conv_module import GraphConvBlock
from .time_pooling import TimePooling
from .. import config as cfg


class CRAINet(nn.Module):
    def __init__(
        self,
        img_size=(512, 512),
        enc_dec_layers=4,
        pool_layers=4,
        in_channels=1,
        out_channels=1,
        fusion_img_size=None,
        fusion_enc_layers=None,
        fusion_pool_layers=None,
        fusion_in_channels=0,
        bounds=None,
    ):

        super().__init__()

        self.freeze_enc_bn = False
        self.net_depth = enc_dec_layers + pool_layers

        # initialize channel inputs and outputs and image size for encoder and decoder
        if cfg.n_filters is None:
            enc_conv_configs = init_enc_conv_configs(
                cfg.conv_factor, img_size, enc_dec_layers, pool_layers, in_channels
            )
            dec_conv_configs = init_dec_conv_configs(
                cfg.conv_factor,
                img_size,
                enc_dec_layers,
                pool_layers,
                in_channels,
                out_channels,
            )
        else:
            enc_conv_configs = init_enc_conv_configs_orig(
                img_size, enc_dec_layers, out_channels, cfg.n_filters
            )
            dec_conv_configs = init_dec_conv_configs_orig(
                img_size, enc_dec_layers, out_channels, cfg.n_filters
            )

        if cfg.attention:
            self.attention_depth = fusion_enc_layers + fusion_pool_layers
            attention_enc_conv_configs = init_enc_conv_configs(
                cfg.conv_factor,
                fusion_img_size,
                fusion_enc_layers,
                fusion_pool_layers,
                fusion_in_channels,
            )

            attention_layers = []
            for i in range(self.attention_depth):
                if i < fusion_enc_layers:
                    kernel = (5, 5)
                else:
                    kernel = (3, 3)
                attention_layers.append(
                    AttentionEncoderBlock(
                        conv_config=attention_enc_conv_configs[i],
                        kernel=kernel,
                        stride=(2, 2),
                        activation=nn.ReLU(),
                    )
                )

                # adjust skip channels for decoder
                if i != self.attention_depth - 1:
                    dec_conv_configs[i]["out_channels"] += attention_enc_conv_configs[
                        self.attention_depth - i - 1
                    ]["in_channels"]
                dec_conv_configs[i]["skip_channels"] += (
                    cfg.skip_layers
                    * attention_enc_conv_configs[self.attention_depth - i - 1][
                        "in_channels"
                    ]
                )
                dec_conv_configs[i]["in_channels"] += attention_enc_conv_configs[
                    self.attention_depth - i - 1
                ]["out_channels"]

            self.attention_module = nn.ModuleList(attention_layers)

        elif fusion_img_size is not None:
            self.channel_fusion_depth = fusion_enc_layers + fusion_pool_layers
            print(
                "in channels 1: ",
                enc_conv_configs[self.net_depth - self.channel_fusion_depth][
                    "in_channels"
                ],
            )
            enc_conv_configs[self.net_depth - self.channel_fusion_depth][
                "in_channels"
            ] += fusion_in_channels
            print(
                "in channels 2: ",
                enc_conv_configs[self.net_depth - self.channel_fusion_depth][
                    "in_channels"
                ],
            )
            dec_conv_configs[self.channel_fusion_depth - 1]["skip_channels"] += (
                cfg.skip_layers * fusion_in_channels
            )

        # define encoding layers
        encoding_layers = []
        for i in range(0, self.net_depth):
            encoding_layers.append(
                EncoderBlock(
                    conv_config=enc_conv_configs[i],
                    kernel=enc_conv_configs[i]["kernel"],
                    stride=(2, 2),
                    activation=nn.ReLU(),
                )
            )
        self.encoder = nn.ModuleList(encoding_layers)

        # define decoding layers
        decoding_layers = []
        for i in range(self.net_depth):
            if i == self.net_depth - 1:
                activation = None
                bias = True
            else:
                activation = nn.LeakyReLU()
                bias = False
            decoding_layers.append(
                DecoderBlock(
                    conv_config=dec_conv_configs[i],
                    kernel=dec_conv_configs[i]["kernel"],
                    stride=(1, 1),
                    activation=activation,
                    bias=bias,
                )
            )
        self.decoder = nn.ModuleList(decoding_layers)

        # optional temporal pooling: collapse the input time window to a
        # single (daily) output field directly inside the model
        self.time_pooling = getattr(cfg, "time_pooling", "none")
        if self.time_pooling != "none":
            self.time_pool = TimePooling(self.time_pooling)

        # optional graph convolution (GNN) layers at the bottleneck
        self.gnn_layers = getattr(cfg, "gnn_layers", 0)
        if self.gnn_layers:
            gnn_channels = enc_conv_configs[self.net_depth - 1]["out_channels"]
            self.gnn = nn.ModuleList(
                [
                    GraphConvBlock(gnn_channels, getattr(cfg, "gnn_neighbors", 8))
                    for _ in range(self.gnn_layers)
                ]
            )

        self.binder = constrain_bounds(bounds)

        # optional distributional head for selected target variables; when
        # disabled (default) the output is a single bounds-scaled channel per
        # variable, exactly as before
        data_types = getattr(cfg, "data_types", None)
        target_types = data_types[-out_channels:] if data_types else []
        self.hurdle_types = [
            t for t in (getattr(cfg, "hurdle_data_types", None) or []) if t in target_types
        ]
        if self.hurdle_types:
            self.hurdle_layout = hurdle_output_layout(target_types, self.hurdle_types)
            self.hurdle_head = HurdleGammaHead(
                dec_conv_configs[self.net_depth - 1]["out_channels"], len(self.hurdle_types)
            )
        else:
            self.hurdle_layout = None
            self.hurdle_head = None

    def _apply_hurdle_head(self, h, coarse_up=None):
        # h: (B, T, C, H, W) decoder output with C == out_channels; returns
        # (B, T, C_out, H, W) where each hurdle variable occupies 3 channels
        # (occurrence logit, gamma shape, conditional mean) and plain
        # variables keep their bounds-scaled channel
        b, t, c, hh, ww = h.shape
        n_out_total = self.hurdle_layout[-1][2] + self.hurdle_layout[-1][3]
        out = h.new_zeros(b, t, n_out_total, hh, ww)
        for target_idx, is_hurdle, out_start, n_out in self.hurdle_layout:
            if not is_hurdle:
                ch = h[:, :, target_idx:target_idx + 1]
                ch = self.binder.binders[target_idx](ch, *self.binder.params[target_idx])
                out[:, :, out_start:out_start + 1] = ch
        x = h.reshape(b * t, c, hh, ww)
        logit_p, k, mu = self.hurdle_head(x, coarse_up=coarse_up)
        j = 0
        for target_idx, is_hurdle, out_start, n_out in self.hurdle_layout:
            if is_hurdle:
                triple = torch.cat([logit_p[:, j:j + 1], k[:, j:j + 1], mu[:, j:j + 1]], dim=1)
                out[:, :, out_start:out_start + 3] = triple.reshape(b, t, 3, hh, ww)
                j += 1
        return out

    def forward(self, input, input_mask, coarse_up=None):
        # create lists for skip connections
        # We split the inputs in case we use the attention module with different image dimension
        h_index = cfg.n_channel_steps
        hs = [input[:, :, :h_index, :, :]]
        hs_mask = [input_mask[:, :, :h_index, :, :]]
        recurrent_states = []

        fusion_input = input[:, :, h_index:, :, :]
        fusion_input_mask = input_mask[:, :, h_index:, :, :]
        h_fusion = fusion_input
        h_fusion_mask = fusion_input_mask
        hs_fusion = []
        hs_fusion_mask = []
        recurrent_fusion_states = []

        # forward pass encoding layers
        for i in range(self.net_depth):
            if h_fusion.size()[1] != 0 and hs[i].shape[3] == h_fusion.shape[3]:
                if not cfg.attention:
                    hs[i] = torch.cat([hs[i], h_fusion], dim=2)
                    hs_mask[i] = torch.cat([hs_mask[i], h_fusion_mask], dim=2)

            h, h_mask, recurrent_state = self.encoder[i](hs[i], hs_mask[i], None)

            # execute attention module if configured
            if cfg.attention and i >= (self.net_depth - self.attention_depth):
                attention_index = i - (self.net_depth - self.attention_depth)
                h_fusion, h_fusion_mask, attention_recurrent_state, attention = (
                    self.attention_module[attention_index](
                        h_fusion, h_fusion_mask, None, h
                    )
                )
                hs_fusion.append(attention)
                hs_fusion_mask.append(h_fusion_mask)
                recurrent_fusion_states.append(attention_recurrent_state)

            # save hidden states for skip connections
            hs.append(h)
            recurrent_states.append(recurrent_state)
            hs_mask.append(h_mask)

        # concat attentions
        if cfg.attention:
            hs[self.net_depth - self.attention_depth] = torch.cat(
                [hs[self.net_depth - self.attention_depth], fusion_input], dim=2
            )
            hs_mask[self.net_depth - self.attention_depth] = torch.cat(
                [hs_mask[self.net_depth - self.attention_depth], fusion_input_mask],
                dim=2,
            )
            for i in range(self.attention_depth):
                hs[i + (self.net_depth - self.attention_depth) + 1] = torch.cat(
                    [hs[i + (self.net_depth - self.attention_depth) + 1], hs_fusion[i]],
                    dim=2,
                )
                hs_mask[i + (self.net_depth - self.attention_depth) + 1] = torch.cat(
                    [
                        hs_mask[i + (self.net_depth - self.attention_depth) + 1],
                        hs_fusion_mask[i],
                    ],
                    dim=2,
                )

                if cfg.lstm_steps:
                    recurrent_state_h, recurrent_state_c = recurrent_states[
                        i + (self.net_depth - self.attention_depth)
                    ]
                    recurrent_fusion_state_h, recurrent_fusion_state_c = (
                        recurrent_fusion_states[i]
                    )
                    recurrent_state_h = torch.cat(
                        [recurrent_state_h, recurrent_fusion_state_h], dim=1
                    )
                    recurrent_state_c = torch.cat(
                        [recurrent_state_c, recurrent_fusion_state_c], dim=1
                    )
                    recurrent_states[i + (self.net_depth - self.attention_depth)] = (
                        recurrent_state_h,
                        recurrent_state_c,
                    )
                elif cfg.gru_steps:
                    recurrent_state_h = recurrent_states[
                        i + (self.net_depth - self.attention_depth)
                    ]
                    recurrent_fusion_state_h = recurrent_fusion_states[i]
                    recurrent_state_h = torch.cat(
                        [recurrent_state_h, recurrent_fusion_state_h], dim=1
                    )
                    recurrent_states[i + (self.net_depth - self.attention_depth)] = (
                        recurrent_state_h
                    )

        # # reverse all hidden states
        # if cfg.recurrent_steps:
        #     for i in range(self.net_depth):
        #         hs[i] = torch.flip(hs[i], (1,))
        #         hs_mask[i] = torch.flip(hs_mask[i], (1,))

        # collapse the time window to a single output field (e.g. daily)
        if self.time_pooling != "none" and cfg.recurrent_steps:
            for i in range(len(hs)):
                hs[i] = self.time_pool(hs[i])
                hs_mask[i] = hs_mask[i].mean(dim=1, keepdim=True)

        h, h_mask = hs[self.net_depth], hs_mask[self.net_depth]

        # apply GNN layers at the bottleneck
        if self.gnn_layers:
            batch_size = h.shape[0]
            h = sequence_to_batch(h)
            h_mask = sequence_to_batch(h_mask)
            for gnn_block in self.gnn:
                h = gnn_block(h, h_mask)
            h = batch_to_sequence(h, batch_size)
            h_mask = batch_to_sequence(h_mask, batch_size)

        # forward pass decoding layers
        for i in range(self.net_depth):
            if cfg.recurrent_steps:
                h, h_mask, recurrent_state = self.decoder[i](
                    h,
                    hs[self.net_depth - i - 1],
                    h_mask,
                    hs_mask[self.net_depth - i - 1],
                    recurrent_states[self.net_depth - 1 - i],
                )
            else:
                h, h_mask, recurrent_state = self.decoder[i](
                    h,
                    hs[self.net_depth - i - 1],
                    h_mask,
                    hs_mask[self.net_depth - i - 1],
                    None,
                )

        if self.hurdle_head is not None:
            h = self._apply_hurdle_head(h, coarse_up=coarse_up)
        else:
            h = self.binder.scale(h)

        # return last element of output from last decoding layer
        return h

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_enc_bn:
            for i in range(self.net_depth):
                if hasattr(self.encoder[i].partial_conv, "bn"):
                    if isinstance(self.encoder[i].partial_conv.bn, nn.BatchNorm2d):
                        self.encoder[i].eval()
