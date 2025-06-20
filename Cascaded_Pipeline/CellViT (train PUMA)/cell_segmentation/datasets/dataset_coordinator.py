from typing import Callable

from torch.utils.data import Dataset
from cell_segmentation.datasets.conic import CoNicDataset
from cell_segmentation.datasets.pannuke import PanNukeDataset
from cell_segmentation.datasets.puma import PumaDataset  # Import PumaDataset

def select_dataset(
    dataset_name: str, split: str, dataset_config: dict, transforms: Callable = None
) -> Dataset:
    """Select a cell segmentation dataset from the provided ones, including PanNuke, CoNIC, and PUMA.

    Args:
        dataset_name (str): Name of dataset to use.
            Must be one of: [pannuke, conic, puma]
        split (str): Split to use.
            Must be one of: ["train", "val", "validation", "test"]
        dataset_config (dict): Dictionary with dataset configuration settings
        transforms (Callable, optional): PyTorch Image and Mask transformations. Defaults to None.

    Raises:
        NotImplementedError: Unknown dataset

    Returns:
        Dataset: Cell segmentation dataset
    """
    assert split.lower() in ["train", "val", "validation", "test"], "Unknown split type!"

    if split == "train":
        folds = dataset_config["train_folds"]
    if split == "val" or split == "validation":
        folds = dataset_config["val_folds"]
    if split == "test":
        folds = dataset_config["test_folds"]
    dataset_path = dataset_config["dataset_path"]
    cache_dataset = dataset_config.get("cache_dataset", False)

    if dataset_name.lower() == "pannuke":
        dataset = PanNukeDataset(
            dataset_path=dataset_path,
            folds=folds,
            transforms=transforms,
            stardist=dataset_config.get("stardist", False),
            regression=dataset_config.get("regression_loss", False),
        )
    elif dataset_name.lower() == "conic":
        dataset = CoNicDataset(
            dataset_path=dataset_path,
            folds=folds,
            transforms=transforms,
            stardist=dataset_config.get("stardist", False),
            regression=dataset_config.get("regression_loss", False),
        )
    elif dataset_name.lower() == "puma":
        dataset = PumaDataset(
            dataset_path=dataset_path,
            folds=folds,
            transforms=transforms,
            cache_dataset=cache_dataset,
        )
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset_name}")
    
    return dataset
