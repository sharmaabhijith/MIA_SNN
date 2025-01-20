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
from collections import Counter

dataset = "cifar10"
model = "vgg16"
ref_models = 4

path = f"datasets/{dataset}"
if os.path.exists(f"{path}.pkl"):
    with open(f"{path}.pkl", "rb") as file:
        all_data = pickle.load(file)
    print(f"Dataset: {dataset} loaded")
else:
    print(f"Dataset: {dataset} pickle file does not exists!")

all_images = all_data.data
all_targets = all_data.targets

print(f"Size of dataset: {len(all_targets)}")

primary_model_path = os.path.join("saved_models", dataset, model, f"ref_models_{ref_models}")
data_split_file = os.path.join(primary_model_path, "data_splits.pkl")
if os.path.exists(data_split_file):
    with open(data_split_file, 'rb') as file:
        data_split_info = pickle.load(file)
    print(f"Data Split File: {data_split_file} loaded")
else:
    print(f"Data Split File: {data_split_file} file does not exists!")


def get_class_distribution(targets, indices):
    subset_targets = targets[indices]
    unsorted_dict = dict(Counter(subset_targets))
    return dict(sorted(unsorted_dict.items()))

for idx, split in enumerate(data_split_info):
    train_distribution = get_class_distribution(all_targets, split["train"])
    test_distribution = get_class_distribution(all_targets, split["test"])

    print(f"Dataset Split {idx}:")
    print(f"Train Class Distribution: {train_distribution}")
    print(f"Test Class Distribution: {test_distribution}")
    print("-" * 50)


