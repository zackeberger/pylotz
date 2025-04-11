from typing import Optional, List

import torch
import medmnist
from torchvision import transforms

_constructors = {
    "OrganMNIST3D": medmnist.OrganMNIST3D
}

def dataset_builder(
    dataset: str,
    split: str = "train",
    normalize: Optional[transforms.Normalize] = None,
    transform: Optional[List[transforms.Compose]] = None,
    path: Optional[str] = None,
):
    """
    Build a medmnist dataset.
    """
    # TODO(zberger): Add support for normalization.
    if normalize is not None:
        raise NotImplementedError(
            "Normalization not implemented for OrganMNIST3D yet."
        )

    # Get the dataset class from medmnist
    if dataset not in _constructors:
        raise ValueError(f"Unknown dataset {dataset}. Supported: {_constructors.keys()}")
    DatasetClass = _constructors[dataset]

    # Instantiate the medmnist dataset, downloading it if necessary.
    dataset_instance = DatasetClass(
        root=path,
        split=split,
        download=True,
        transform=transform,
    )

    return dataset_instance


def OrganMNIST3D(
    split: str = "train",
    path: Optional[str] = None,
    cast_to_float: bool = True,
    norm: bool = False,
    augmentation: bool = False,
    augment_kw: dict = None
):
    """
    Thin wrapper around medmnist.OrganMNIST3D.
    """
    # TODO(zberger): Add support for normalization and augmentations (transform).
    normalize = None
    if norm or augmentation or augment_kw is not None:
        raise NotImplementedError(
            "Normalization and augmentation not implemented for OrganMNIST3D yet."
        )

    # Build a transform that converts the sample to float.
    transform = None
    if cast_to_float:
        transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.tensor(x) if not isinstance(x, torch.Tensor) else x),
            transforms.Lambda(lambda x: x.float())
        ])

    dataset = dataset_builder("OrganMNIST3D", split, normalize, transform, path)
    dataset.shape = (1,28,28,28)
    dataset.n_classes = 11
    return dataset
