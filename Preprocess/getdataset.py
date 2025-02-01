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
from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100
from .getdataloader import CIFAR10Policy, Cutout
from utils import GlobalLogger


logger = GlobalLogger.get_logger(__name__)

def get_dataset(dataset: str) -> Any:
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
    data_path = f"datasets/{dataset}"
    if os.path.exists(f"{data_path}.pkl"):
        with open(f"{data_path}.pkl", "rb") as file:
            all_data = pickle.load(file)
        if logger:
            logger.info(f"Load data from {data_path}.pkl")
        else:
            print(f"Load data from {data_path}.pkl")
    else:
        if dataset == "cifar10":
            transform = transforms.Compose([transforms.ToTensor()])
            all_data = CIFAR10(root=data_path, train=True, download=True, transform=transform)
            test_data = CIFAR10(root=data_path, train=False, download=True, transform=transform)
            all_features = np.concatenate([all_data.data, test_data.data], axis=0)
            all_targets = np.concatenate([all_data.targets, test_data.targets], axis=0)
            all_data.data = all_features
            all_data.targets = all_targets
            with open(f"{data_path}.pkl", "wb") as file:
                pickle.dump(all_data, file)
            if logger:
                logger.info(f"Save data to {data_path}.pkl")
            else:
                print(f"Save data to {data_path}.pkl")
        elif dataset == "cifar100":
            transform = transforms.Compose([transforms.ToTensor()])
            all_data = CIFAR100(root=data_path, train=True, download=True, transform=transform)
            test_data = CIFAR100(root=data_path, train=False, download=True, transform=transform)
            all_features = np.concatenate([all_data.data, test_data.data], axis=0)
            all_targets = np.concatenate([all_data.targets, test_data.targets], axis=0)
            all_data.data = all_features
            all_data.targets = all_targets
            with open(f"{data_path}.pkl", "wb") as file:
                pickle.dump(all_data, file)
            if logger:
                logger.info(f"Save data to {data_path}.pkl")
            else:
                print(f"Save data to {data_path}.pkl")
        elif dataset in ("imagenette", "imagewoof"):
            transform = transforms.Compose([transforms.ToTensor()])
            all_data = ImageFolder(os.path.join(data_path, "train"), transform=transform)
            test_data = ImageFolder(os.path.join(data_path, "test"), transform=transform)
            all_features = np.concatenate([all_data.samples, test_data.samples], axis=0)
            all_targets = np.concatenate([all_data.targets, test_data.targets], axis=0)
            all_data.data = all_features
            all_data.targets = all_targets
            with open(f"{data_path}.pkl", "wb") as file:
                pickle.dump(all_data, file)
            if logger:
                logger.info(f"Save data to {data_path}.pkl")
            else:
                print(f"Save data to {data_path}.pkl")
        else:
            raise NotImplementedError(f"{dataset} is not implemented")
    if logger:
        logger.info(f"The whole dataset size: {len(all_data)}")
    else:
        print(f"The whole dataset size: {len(all_data)}")
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
    master_keep = np.full((2*num_splits - 1, dataset_size), True, dtype=bool)

    for i in range(num_splits):
        np.random.shuffle(indices)
        master_keep[i * 2, indices[split_index:]] = False
        keep = master_keep[i * 2, :]
        train_indices = np.where(keep)[0]
        test_indices = np.where(~keep)[0]
        data_splits.append(
            {
                "train": train_indices,
                "test": test_indices,
            }
        )
        if i==0:
            continue
        data_splits.append(
            {
                "train": test_indices,
                "test": train_indices,
            }
        )

    return data_splits


class TransformDataset(Dataset):
    def __init__(self, dataset_name, dataset, train=True):
        """
        Args:
            dataset (Dataset): Existing PyTorch dataset with data and labels.
            train (callable, optional): Whether to transform train or test data
        """
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.train = train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get data and label from the original dataset
        data, label = self.dataset[idx]
        # Apply transformation to the data
        if self.dataset_name=="cifar10":
            train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                # Add Cutout or other custom transforms if needed
            ])
            test_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        elif self.dataset_name=="cifar100":
            train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]]),
                Cutout(n_holes=1, length=8)
            ])
            test_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]]),
            ])
        elif self.dataset_name in ("imagenette", "imagewoof"):
            train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            test_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            print(f"No dataset transform defined for: {self.dataset_name}")

        if self.train:
            data = train_transform(data)
        else:
            data = test_transform(data)
        
        return data, label
    

