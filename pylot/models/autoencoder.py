from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..nn import ConvBlock, get_nonlinearity

class Flatten(nn.Module):
    """
    Flattens [B, C, D, H, W, ...] -> [B, C*D*H*W*...].
    """
    def forward(self, x):
        return torch.flatten(x, start_dim=1)

class Unflatten(nn.Module):
    """
    Unflattens [B, C] -> [B, C, 1, ...].
    """
    def __init__(self, dims: Literal[1, 2, 3]):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        # x is [B, C], reshape to [B, C, 1, ...]
        b, c = x.shape[0], x.shape[1]
        return x.view(b, c, *([1]*self.dims))

@dataclass(eq=False, repr=False)
class Autoencoder(nn.Module):
    in_channels: int
    out_channels: int
    filters: List[int]
    latent_dim: int = 128
    up_filters: Optional[List[int]] = None
    bottleneck_activation: Optional[str] = "ReLU"
    out_activation: Optional[str] = None
    convs_per_block: int = 1
    dims: Literal[1, 2, 3] = 2
    interpolation_mode: str = "linear"
    conv_kws: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        super().__init__()

        filters = list(self.filters)
        if self.up_filters is None:
            self.up_filters = filters[::-1]
        assert len(self.up_filters) == len(self.filters)

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        conv_args = dict(dims=self.dims)
        if self.conv_kws:
            conv_args.update(self.conv_kws)

        # Build the encoder
        for in_ch, out_ch in zip([self.in_channels] + filters[:-1], filters):
            c = ConvBlock(in_ch, [out_ch] * self.convs_per_block, **conv_args)
            self.down_blocks.append(c)
        
        # Add final layer to collapse spatial dimension in encoder
        pool_cls = [nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d][self.dims - 1]
        self.global_pool = pool_cls((1,)*self.dims) # e.g. (1,1,1) in 3D

        # Build the bottleneck
        #
        # The number of channels at the bottom of the UNet is filters[-1], so
        # after pooling we have shape, e.g. in 3D, (B, filters[-1], 1, 1, 1).
        bottleneck_in = filters[-1]
        self.flatten = Flatten()
        self.bottleneck_fc1 = nn.Linear(bottleneck_in, self.latent_dim)
        self.bottleneck_fc2 = nn.Linear(self.latent_dim, bottleneck_in)
        self.bottleneck_act = get_nonlinearity(self.bottleneck_activation)()
        self.unflatten = Unflatten(self.dims)

        # Build the decoder
        prev_out_ch = filters[-1]
        for out_ch in self.up_filters:
            in_ch = prev_out_ch
            c = ConvBlock(in_ch, [out_ch] * self.convs_per_block, **conv_args)
            prev_out_ch = out_ch
            self.up_blocks.append(c)

        # Final output convolution
        # filters[0] == up_filters[-1]
        self.out_conv = ConvBlock(
            prev_out_ch,
            [self.out_channels],
            activation=None,
            kernel_size=1,
            dims=self.dims,
            norm=None,
        )

        if self.interpolation_mode == "linear":
            self.interpolation_mode = ["linear", "bilinear", "trilinear"][self.dims - 1]

        if self.out_activation:
            if self.out_activation == "Softmax":
                # For Softmax, we need to specify the channel dimension
                self.out_fn = nn.Softmax(dim=1)
            else:
                self.out_fn = get_nonlinearity(self.out_activation)()

        self.reset_parameters()

    def reset_parameters(self):
        for group in (self.down_blocks, self.up_blocks, [self.out_conv]):
            for module in group:
                module.reset_parameters()
    
    # TODO(zberger): refactor forward pass to use this function for
    # code readability.
    def encode(self, x: Tensor) -> Tensor:
        pool_fn = [F.max_pool1d, F.max_pool2d, F.max_pool3d][self.dims - 1]
        for i, conv_block in enumerate(self.down_blocks):
            x = conv_block(x)
            if i < len(self.down_blocks) - 1:
                x = pool_fn(x, 2)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.bottleneck_fc1(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        pool_fn = [F.max_pool1d, F.max_pool2d, F.max_pool3d][self.dims - 1]

        # Encoder: down blocks
        conv_outputs = []
        for i, conv_block in enumerate(self.down_blocks):
            x = conv_block(x)
            conv_outputs.append(x)
            # Only pool between blocks, not after the last
            if i < len(self.down_blocks) - 1:
                x = pool_fn(x, 2)

        # Force final shape to [B, filters[-1], 1,1,1]
        x = self.global_pool(x)

        # Bottleneck
        x = self.flatten(x)
        x = self.bottleneck_fc1(x)
        x = self.bottleneck_act(x)
        x = self.bottleneck_fc2(x)
        x = self.bottleneck_act(x)
        x = self.unflatten(x)

        # Decoder: up blocks
        for i, conv_block in enumerate(self.up_blocks, start=1):
            target_shape = conv_outputs[-i].size()[-self.dims:]
            x = F.interpolate(
                x,
                size=target_shape,
                align_corners=True,
                mode=self.interpolation_mode,
            )
            x = conv_block(x)

        x = self.out_conv(x)
        if self.out_activation:
            x = self.out_fn(x)

        return x
