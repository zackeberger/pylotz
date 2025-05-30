import torch
import monai
import numpy as np
from pylot.experiment.util import eval_config
from typing import List


class TransformPipeline:
    """Pipeline that applies 3D MONAI array-based transforms.
       If `data` is 5D (B, C, D, H, W), we loop over each item in the batch
       since MONAI expects 4D (C, D, H, W) tensors. If `data` is 4D (C, D, H, W),
       we apply the transforms directly.

       Note: We convert from PyTorch (C, D, H, W) to MONAI (C, H, W, D) format
       before applying transforms, then convert back.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        # If we detect a 5D tensor => [B, C, D, H, W]
        if data.ndim == 5:
            # MONAI expects (C, H, W, D), so we'll apply transforms to each sample
            # individually. Additionally, we will permute the axes from (D, H, W)
            # to (H, W, D) before applying transforms, and permute them back afterward.
            # We use .contiguous() because permute returns a non-contiguous view,
            # while we need a contiguous tensor for later processing.
            out_list = []
            for i in range(data.shape[0]):
                sample = data[i]  # shape (C, D, H, W)
                sample = sample.permute(0, 2, 3, 1)  # (C, D, H, W) -> (C, H, W, D)
                for t in self.transforms:
                    sample = t(sample)
                sample = sample.permute(0, 3, 1, 2).contiguous()  # (C, H, W, D) -> (C, D, H, W)
                out_list.append(sample)
            # Stack them back up: (B, C, D, H, W)
            return torch.stack(out_list, dim=0)
        else:
            # For a single sample => (C, D, H, W)
            data = data.permute(0, 2, 3, 1)  # (C, D, H, W) -> (C, H, W, D)
            for t in self.transforms:
                data = t(data)
            data = data.permute(0, 3, 1, 2).contiguous()  # (C, H, W, D) -> (C, D, H, W)
            return data

def build_3d_augmentations(pipeline_cfg: List[dict]):
    """
    Build a pipeline of 3D transforms.
    """
    transforms = []
    for transform_spec in pipeline_cfg:
        transform = eval_config(transform_spec)
        transforms.append(transform)

    return TransformPipeline(transforms)


def RandomAffine(
    prob=1.0,
    scale_range=(0, 0, 0),
    rotate_range=(0, 0, 0),
    translate_range=(0, 0, 0),
    padding_mode="border"
):
    return monai.transforms.RandAffine(
        prob=prob,
        scale_range=scale_range,
        rotate_range=rotate_range,
        translate_range=translate_range,
        padding_mode=padding_mode
    )

def RandomContrastAdjustment(
    prob=1.0,
    gamma_range=(0.5, 2.0)
):
    return monai.transforms.RandAdjustContrast(
        prob=prob,
        gamma=gamma_range
    )
