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
    """
    Flexible convolutional autoencoder supporting optional spatial collapsing
    and fully-connected bottlenecks.

    This class builds a symmetric encoder-decoder architecture with optional 
    flattening and projection at the bottleneck. It is designed for 1D, 2D, or 
    3D data and supports residual blocks, configurable convolutional depth, and
    flexible latent bottleneck strategies.

    Architecture:
        - Encoder: A stack of ConvBlocks with optional downsampling between them.
        - Bottleneck:
            - If collapse_spatial=True: Applies AdaptiveMaxPool to reduce spatial 
              dimensions to 1 (e.g., [B, C, 1, 1, 1]).
            - If fc_bottleneck=True: Further flattens and projects to a latent_dim
              vector using Linear layers. If fc_bottleneck is False, then the
              latent_dim is ignored and the number of channels is effectively
              the latent dimension.
        - Decoder: A mirror of the encoder using ConvBlocks and upsampling via 
          interpolation.

    Key Options:
        collapse_spatial (bool):
            If True, applies global pooling to collapse spatial dimensions at the 
            bottom of the encoder. This results in a fixed-size spatial bottleneck 
            tensor of shape [B, C, 1, ..., 1].
            
            - If True and fc_bottleneck=False:
                The latent representation is [B, C], returned from encode().
                In forward(), shape becomes [B, C, 1, ..., 1] before decoding.
            - If False:
                No spatial reduction is performed, and feature maps remain spatially
                structured throughout the bottleneck.

        fc_bottleneck (bool):
            If True, applies two fully-connected layers at the bottleneck to compress
            the [B, C] vector to latent_dim and then project back to [B, C].
            Only valid if collapse_spatial is also True.

    Example configurations:
        - Vanilla convolutional autoencoder:
            collapse_spatial=False, fc_bottleneck=False

        - Autoencoder with spatial bottleneck only:
            collapse_spatial=True, fc_bottleneck=False

        - Autoencoder with spatial + fc bottleneck (SimCLR-style):
            collapse_spatial=True, fc_bottleneck=True
    
    Parameters:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        filters: List of channel dimensions for encoder (decoder is mirrored).
        latent_dim: Size of latent projection if using fc_bottleneck.
        collapse_spatial: Whether to apply global spatial pooling at bottleneck.
        fc_bottleneck: Whether to add fully-connected layers at bottleneck.
        up_filters: Optional custom list of decoder filters (reversed by default).
        bottleneck_activation: Activation to use between linear layers.
        out_activation: Optional nonlinearity to apply to final output.
        convs_per_block: Number of convolutions per ConvBlock.
        dims: Dimensionality of data (1D, 2D, or 3D).
        interpolation_mode: Interpolation method for upsampling.
        conv_kws: Extra kwargs passed to ConvBlock (e.g., residual=True).
    """
    in_channels: int
    out_channels: int
    filters: List[int]
    latent_dim: int = 128
    collapse_spatial: bool = True
    fc_bottleneck:  bool = False
    up_filters: Optional[List[int]] = None
    bottleneck_activation: Optional[str] = "ReLU"
    out_activation: Optional[str] = None
    convs_per_block: int = 1
    dims: Literal[1, 2, 3] = 2
    interpolation_mode: str = "linear"
    conv_kws: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        super().__init__()

        if self.fc_bottleneck:
            # A FC bottleneck only makes sense if we pooled to (1, …, 1)
            assert (
                self.collapse_spatial
            ), "`fc_bottleneck=True` requires `collapse_spatial=True`"

        filters = list(self.filters)
        if self.up_filters is None:
            self.up_filters = filters[::-1]
        assert len(self.up_filters) == len(self.filters), "`up_filters` length mismatch"

        conv_args = dict(dims=self.dims, **(self.conv_kws or {}))

        # ---------------- Encoder (down path) ---------------- #
        self.down_blocks = nn.ModuleList()
        for in_ch, out_ch in zip([self.in_channels] + filters[:-1], filters):
            c = ConvBlock(in_ch, [out_ch] * self.convs_per_block, **conv_args)
            self.down_blocks.append(c)
        
        # Optional spatial collapse
        if self.collapse_spatial:
            pool_cls = [nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d][self.dims - 1]
            self.global_pool = pool_cls((1,) * self.dims)
        else:
            self.global_pool = nn.Identity()
 
        # ---------------- Bottleneck ---------------- #
        # If spatial collapse is set we will need to flatten the output. We're
        # just going to set it here, since there's no harm in having it.
        self.flatten = Flatten()
        # The number of channels at the bottom of the UNet is filters[-1], so
        # after pooling we have shape, e.g. in 3D, (B, filters[-1], 1, 1, 1).
        bottleneck_in = filters[-1]
        if self.collapse_spatial and self.fc_bottleneck:
            self.bottleneck_fc1 = nn.Linear(bottleneck_in, self.latent_dim)
            self.bottleneck_fc2 = nn.Linear(self.latent_dim, bottleneck_in)
            self.bottleneck_act = get_nonlinearity(self.bottleneck_activation)()
            self.unflatten = Unflatten(self.dims)
        
        # ---------------- Decoder (up path) ---------------- #
        self.up_blocks = nn.ModuleList()
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

    def reset_parameters(self) -> None:
        """Re‑initialise weights of all learnable modules."""
        for group in (self.down_blocks, self.up_blocks, [self.out_conv]):
            for m in group:
                m.reset_parameters()
        if self.fc_bottleneck:
            self.bottleneck_fc1.reset_parameters()
            self.bottleneck_fc2.reset_parameters()
    
    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        pool_fn = [F.max_pool1d, F.max_pool2d, F.max_pool3d][self.dims - 1]
        for i, conv_block in enumerate(self.down_blocks):
            x = conv_block(x)
            if i < len(self.down_blocks) - 1:
                x = pool_fn(x, 2)
        if self.collapse_spatial:
            x = self.global_pool(x)
            x = self.flatten(x)
            if self.fc_bottleneck:
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

        # Optional: spatial collapse
        if self.collapse_spatial:
            x = self.global_pool(x)

        # Optional: FC Bottleneck after spatial collapse
        if self.collapse_spatial and self.fc_bottleneck:
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
