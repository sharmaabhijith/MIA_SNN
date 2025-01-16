import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from spiking_layer_ours import *  # Import custom spiking layer module
from Models import modelpool  # Import function to fetch models
from Preprocess import datapool, get_dataloader_from_dataset, load_dataset, split_dataset  # Import data preprocessing module
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import Subset
import os
import argparse
from funcs import *  # Import additional custom functions
import numpy as np
import time
import pickle
import logging
from utils import *  # Import utility functions

# Parse command-line arguments
parser = argparse.ArgumentParser(description='PyTorch ANN-SNN Conversion')

# Define arguments for model parameters and settings
parser.add_argument('--t', default=300, type=int, help='T Latency length (Simulation time-steps)')
parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset name',
                    choices=['cifar10', 'cifar100', 'tiny-imagenet', 'imagenet', 'fashion'])
parser.add_argument('--model', default='vgg16', type=str, help='Model name',
                    choices=['small', 'vgg16', 'resnet18', 'resnet20', 'vgg16_no_bn',
                             'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg16_normed', 'alexnet',
                             'resnet18', 'resnet19', 'resnet20', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'cifarnet'])
parser.add_argument('--checkpoint', default='./saved_models', type=str, help='Directory for saving models')
parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
parser.add_argument('--wd', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--reference_models', default=4, type=int, help='Number of reference models')

args = parser.parse_args()

# Check device configuration and set accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters
input_size = 784
hidden_size = 500
num_classes = 10  # Number of output classes
num_epochs = args.epochs
batch_size = 256  

# Creating directory to save trained models and their logs
primary_model_path = os.path.join(args.checkpoint, args.dataset, args.model, f"ref_models_{args.reference_models}")
primary_log_path = os.path.join("logs", args.dataset, args.model, f"ref_models_{args.reference_models}")

for model_idx in range(0, args.reference_models+1):
    # Model created with idx 0 is always the target model
    # Create model dir
    full_model_path = os.path.join(primary_model_path, f"model_{model_idx}")
    if os.path.exists(full_model_path) is False:
        print("Creating model directory:", full_model_path)
        os.makedirs(full_model_path)
    # Create log dir
    full_log_path = os.path.join(primary_log_path, f"model_{model_idx}")
    if os.path.exists(full_log_path) is False:
        print("Creating log directory:", full_log_path)
        os.makedirs(full_log_path)

    # Configure logging
    GlobalLogger.initialize(
        os.path.join(
            primary_log_path, 
            f"model_{model_idx}", 
            f"ann_lr{args.lr}.log"
        )
    )
    logger = GlobalLogger.get_logger(__name__)
    # Log initial configuration details
    logger.info("Starting ANN Training")
    logger.info(f"Arguments: {args}")
    logger.info(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    if model_idx==0:
        # Load the dataset using the specified parameters
        logger.info("Loading dataset...")
        dataset = load_dataset(args.dataset, logger)
        data_split_info = split_dataset(len(dataset), args.reference_models)
        logger.info(f"Total datasets (train-test pairs): {len(data_split_info)}. Assert should be target + reference models: {1 + args.reference_models}")
        pickle.dump(data_split_info, open(os.path.join(primary_model_path, "data_splits.pkl"), "wb"))
    # Creating dataloader
    train_idxs = data_split_info[model_idx]["train"]
    test_idxs = data_split_info[model_idx]["test"]
    logger.info(
        f"Training model {model_idx}: Train size {len(train_idxs)}, Test size {len(test_idxs)}"
    )
    logger.info("Creating dataloader...")
    train_loader = get_dataloader_from_dataset(Subset(dataset, train_idxs), batch_size=batch_size, train=True)
    test_loader = get_dataloader_from_dataset(Subset(dataset, test_idxs), batch_size=batch_size, train=False)
    logger.info(f"Dataset loaded successfully. Training batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    # Load the specified model from the model pool
    logger.info(f"Loading model: {args.model} for dataset: {args.dataset}")
    model = modelpool(args.model, args.dataset)
    model.to(device)
    savename = os.path.join(primary_model_path, f"model_{model_idx}", f"ann")
    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    # Start training the ANN model
    logger.info("Starting training...")
    train_ann(train_loader, test_loader, model, num_epochs, device, criterion, args.lr, args.wd, savename)
    logger.info("Training completed successfully")

    GlobalLogger.reset_logger()
