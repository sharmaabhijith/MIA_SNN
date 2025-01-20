import os
import math
import sys
import pickle
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from collections import Counter

sys.path.append("/home/ubuntu/ashar/MIA_SNN/")
dataset_names = ["cifar10"]
model_names = ["vgg16", "resnet18", "resnet34"]
ref_models_list = [4]


def indices_in_other_array(target_data, reference_data):
    """
    This function takes target_data index and reference_data index
    and returns the membership of the target train and test data.
    0 for the data in the first half of the reference_data and 
    1 for the data in the second half of the reference_data.
    """
    mapping = {value: index for index, value in enumerate(reference_data)}
    indices = np.array([mapping[element] for element in target_data])
    n = len(target_data)
    membership = np.int64(indices >= n//2 )
    return {"train": membership[:n//2] , "test": membership[n//2:]}


for dataset in dataset_names:
    print(f"Dataset: {dataset}")
    print("="*100)
    for model in model_names:
        print(f"Model: {model}")
        for ref_models in ref_models_list:
            print(f"Refence Models: {ref_models}")
            primary_model_path = os.path.join("../saved_models", dataset, model, f"ref_models_{ref_models}")
            data_split_file = os.path.join(primary_model_path, "data_splits.pkl")
            if os.path.exists(data_split_file):
                with open(data_split_file, 'rb') as file:
                    data_split_info = pickle.load(file)
                print(f"Data Split File: {data_split_file} loaded")
            else:
                print(f"Data Split File: {data_split_file} file does not exists!")
            # Create Memeberships w.r.t Target indexes
            # Target model is always at 0 index
            target_idxs = np.concatenate((data_split_info[0]["train"], data_split_info[0]["test"]), axis=None)
            target_membership_wrt_reference = []
            for idx in range(1, len(data_split_info)):
                ref_idxs = np.concatenate((data_split_info[idx]["train"], data_split_info[idx]["test"]), axis=None)
                membership_dict = indices_in_other_array(target_idxs, ref_idxs)
                target_membership_wrt_reference.append(membership_dict)
            membership_file =  os.path.join(primary_model_path, "memberships.pkl")
            with open(membership_file, "wb") as file:
                pickle.dump(target_membership_wrt_reference, file) 
            print(f"Membership data dumped at: {membership_file}")
            print("-"*100)
    print("="*100)




