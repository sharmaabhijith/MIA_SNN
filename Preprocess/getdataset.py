import math
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from .getdataloader import CIFAR10Policy


def get_dataset(dataset: str, logger: Any, **kwargs: Any) -> Any:
    """
    Function to load the dataset from the pickle file or download it from the internet.

    Args:
        dataset (str): Dataset name.
        data_dir (str): Indicate the log directory for loading the dataset.
        logger (logging.Logger): Logger object for the current run.

    Raises:
        NotImplementedError: If the dataset is not implemented.

    Returns:
        Any: Loaded dataset.
    """
    path = f"datasets/{dataset}"
    if os.path.exists(f"{path}.pkl"):
        with open(f"{path}.pkl", "rb") as file:
            all_data = pickle.load(file)
        logger.info(f"Load data from {path}.pkl")
    else:
        if dataset == "cifar10":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
            all_data = torchvision.datasets.CIFAR10(
                root=path, train=True, download=True, transform=transform
            )
            test_data = torchvision.datasets.CIFAR10(
                root=path, train=False, download=True, transform=transform
            )
            all_features = np.concatenate([all_data.data, test_data.data], axis=0)
            all_targets = np.concatenate([all_data.targets, test_data.targets], axis=0)
            all_data.data = all_features
            all_data.targets = all_targets
            with open(f"{path}.pkl", "wb") as file:
                pickle.dump(all_data, file)
            logger.info(f"Save data to {path}.pkl")
        elif dataset == "cifar100":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
            all_data = torchvision.datasets.CIFAR100(
                root=path, train=True, download=True, transform=transform
            )
            test_data = torchvision.datasets.CIFAR100(
                root=path, train=False, download=True, transform=transform
            )
            all_features = np.concatenate([all_data.data, test_data.data], axis=0)
            all_targets = np.concatenate([all_data.targets, test_data.targets], axis=0)
            all_data.data = all_features
            all_data.targets = all_targets
            with open(f"{path}.pkl", "wb") as file:
                pickle.dump(all_data, file)
            logger.info(f"Save data to {path}.pkl")
        elif dataset == "texas100":
            if os.path.exists(f"{data_dir}/dataset_texas/feats"):
                X = (
                    pd.read_csv(
                        f"{data_dir}/dataset_texas/feats", header=None, encoding="utf-8"
                    )
                    .to_numpy()
                    .astype(np.float32)
                )
                y = (
                    pd.read_csv(
                        f"{data_dir}/dataset_texas/labels",
                        header=None,
                        encoding="utf-8",
                    )
                    .to_numpy()
                    .reshape(-1)
                    - 1
                )
                all_data = TabularDataset(X, y)
                with open(f"{path}.pkl", "wb") as file:
                    pickle.dump(all_data, file)
                logger.info(f"Save data to {path}.pkl")
            else:
                raise NotImplementedError(
                    f"{dataset} is not installed correctly in {data_dir}/dataset_texas"
                )
        else:
            raise NotImplementedError(f"{dataset} is not implemented")

    logger.info(f"The whole dataset size: {len(all_data)}")
    return all_data


def split_dataset_for_training(dataset_size, num_reference_models):
    """
    Split dataset into training and test partitions for model pairs.

    Args:
        dataset_size (int): Total number of samples in the dataset.
        num_reference_models (int): Number of model pairs to be trained, with each pair trained on different halves of the dataset.

    Returns:
        data_split (list): List of dictionaries containing training and test split indices for each model.
        master_keep (np.array): D boolean array indicating the membership of samples in each model's training set.
    """
    data_splits = []
    indices = np.arange(dataset_size)
    split_index = len(indices) // 2
    num_splits = math.ceil(num_reference_models/2) + 1 # Extra 1 for the target model
    master_keep = np.full((2*num_splits, dataset_size), True, dtype=bool)

    for i in range(num_splits):
        np.random.shuffle(indices)
        master_keep[i * 2, indices[split_index:]] = False
        master_keep[i * 2 + 1, indices[:split_index]] = False
        keep = master_keep[i * 2, :]
        train_indices = np.where(keep)[0]
        test_indices = np.where(~keep)[0]
        data_splits.append(
            {
                "train": train_indices,
                "test": test_indices,
            }
        )
        data_splits.append(
            {
                "train": test_indices,
                "test": train_indices,
            }
        )

    return data_splits, master_keep


class TransformDataset(Dataset):
    def __init__(self, dataset, train=True):
        """
        Args:
            dataset (Dataset): Existing PyTorch dataset with data and labels.
            train (callable, optional): Whether to transform train or test data
        """
        self.dataset = dataset
        self.train = train

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def train_transform():
        """Return train transformations."""
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),  # Assuming CIFAR10Policy is defined elsewhere
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            Cutout(n_holes=1, length=16)  # Assuming Cutout is defined elsewhere
        ])

    @staticmethod
    def test_transform():
        """Return test transformations."""
        return transforms.Compose([
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    def __getitem__(self, idx):
        # Get data and label from the original dataset
        data, label = self.dataset[idx]
        # Apply transformation to the data
        if self.train:
            data = TransformDataset.train_transform(data)
        else:
            data = TransformDataset.test_transform(data)
        
        return data, label
    
