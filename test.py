import numpy as np
from torch import nn
import torch
from tqdm import tqdm
from utils import *
import random
import os
import logging
import pickle
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset, Dataset
import torchvision
import torchvision.transforms as transforms
from spiking_layer_ours import *  # Import custom spiking layer module
from Models import modelpool  # Import function to fetch models
from Preprocess import datapool, get_dataloader_from_dataset, load_dataset, split_dataset  # Import data preprocessing module
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import Subset
import argparse
from funcs import *  # Import additional custom functions
import time
from utils import *  # Import utility functions
import warnings
warnings.filterwarnings("ignore")



def eval_ann(test_dataloader, model, loss_fn, device, rank=0):
    logger.info("Starting evaluation...")
    epoch_loss = 0
    tot = torch.tensor(0.).cuda(device)
    model.eval()
    model.cuda(device)
    length = 0
    with torch.no_grad():
        for img, label in test_dataloader:
            img = img.cuda(device)
            label = label.cuda(device)
            out = model(img)
            loss = loss_fn(out, label)
            epoch_loss += loss.item()
            length += len(label)
            tot += (label == out.max(1)[1]).sum().data

    accuracy = tot / length
    avg_loss = epoch_loss / length
    logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
    return accuracy, avg_loss


# ARGUMENTS
MODEL = "vgg16"
DATASET = "cifar10"
CHECKPOINT = "saved_models"
REFERENCE_MODELS = 4
MODEL_IDX = 0
BATCH_SIZE = 64


GlobalLogger.initialize("./test.log")
logger = GlobalLogger.get_logger(__name__)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
primary_model_path = os.path.join(CHECKPOINT, DATASET, MODEL, f"ref_models_{REFERENCE_MODELS}")

dataset = load_dataset(DATASET, logger)
try:
    data_split_file = os.path.join(primary_model_path, "data_splits.pkl")
    with open(data_split_file, 'rb') as file:
        data_split_info = pickle.load(file)
    print("Data split information successfully loaded:")
except FileNotFoundError:
    print(f"Error: The file '{data_split_file}' does not exist")


test_idxs = data_split_info[MODEL_IDX]["test"]
print(f"Testing Model: Test size {len(test_idxs)}")
print("Creating dataloader...")
test_loader = get_dataloader_from_dataset(Subset(dataset, test_idxs), batch_size=BATCH_SIZE, train=False)

model = modelpool(MODEL, DATASET)
model.to(device)
model_path = os.path.join(primary_model_path, f"model_{MODEL_IDX}", "ann")
model.load_state_dict(torch.load(model_path + '.pth'))

criterion = nn.CrossEntropyLoss()

tmp_acc, val_loss = eval_ann(test_loader, model, criterion, device)
print(f"Validation Accuracy: {tmp_acc} | Validation Loss: {val_loss}")