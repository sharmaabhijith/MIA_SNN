import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from spiking_layer_ours import *
from Models import modelpool
from Preprocess import get_dataloader_from_dataset, load_dataset
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import Subset
import os
import argparse
import numpy as np
import calc_th_with_c as ft
from copy import deepcopy
import pickle
from utils import *


print("Initializing ANN-to-SNN Conversion Script...")
parser = argparse.ArgumentParser(description='PyTorch ANN-SNN Conversion')
parser.add_argument('--t', default=300, type=int, help='T Latency length (Simulation time-steps)')
parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset name',
                    choices=['cifar10', 'cifar100', 'imagenette', 'imagewoof'])
parser.add_argument('--model', default='vgg16', type=str, help='Model name',
                    choices=['vgg16', 'resnet18', 'resnet34', 'resnet50'])
parser.add_argument('--checkpoint', default='./saved_models', type=str, help='Directory for saving models')
parser.add_argument('--batchsize', default=64, type=int)
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--wd', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--device', default='cuda:0', type=str)
parser.add_argument('--constant_lr', action='store_true')
parser.add_argument('--reference_models', default=4, type=int, help='Number of reference models')

args = parser.parse_args()

# Check device configuration and set accordingly
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
n_steps = args.t

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
            f"snn_lr{args.lr}_T{n_steps}.log"
        )
    )
    logger = GlobalLogger.get_logger(__name__)
    # Log initial configuration details
    logger.info("Starting SNN calibration")
    logger.info(f"Arguments: {args}")
    logger.info(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    if model_idx==0:
        # Load the dataset using the specified parameters
        logger.info("Loading dataset...")
        dataset = load_dataset(args.dataset)
        try:
            data_split_file = os.path.join(primary_model_path, "data_splits.pkl")
            with open(data_split_file, 'rb') as file:
                data_split_info = pickle.load(file)
            logger.info("Data split information successfully loaded:")
        except FileNotFoundError:
            logger.info(f"Error: The file '{data_split_file}' does not exist")
    # Creating dataloader
    train_idxs = data_split_info[model_idx]["train"]
    test_idxs = data_split_info[model_idx]["test"]
    logger.info(
        f"Training model {model_idx}: Train size {len(train_idxs)}, Test size {len(test_idxs)}"
    )
    logger.info("Creating dataloader...")
    train_loader = get_dataloader_from_dataset(args.dataset, Subset(dataset, train_idxs), batch_size=batch_size, train=True)
    test_loader = get_dataloader_from_dataset(args.dataset, Subset(dataset, test_idxs), batch_size=batch_size, train=False)
    logger.info(f"Dataset loaded successfully. Training batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    # Load the specified model from the model pool
    logger.info(f"Loading model: {args.model} for dataset: {args.dataset}")
    model = modelpool(args.model, args.dataset)
    savename = os.path.join(primary_model_path, f"model_{model_idx}", "ann")
    model.load_state_dict(torch.load(savename + '.pth'))

    num_relu = str(model).count('ReLU')
    thresholds = torch.zeros(num_relu, 2*n_steps)
    thresholds1 = torch.Tensor(np.load('%s_threshold_all_noaug%d.npy' % (savename, 1)))
    ann_to_snn(model, thresholds, thresholds1, n_steps)
    if n_steps > 1:
        model.load_state_dict(torch.load(f"{savename}_snn_T{n_steps-1}.pth"))
    model.to(device)

    # Setting up optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    # para1, para2, para3 = regular_set(model)
    # optimizer = torch.optim.SGD([
    #                            {'params': para1, 'weight_decay': args.wd},
    #                            {'params': para2, 'weight_decay': args.wd},
    #                            {'params': para3, 'weight_decay': args.wd}
    #                            ],
    #                            lr=args.lr, momentum=0.9)
    if args.constant_lr:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, verbose=True)

    logger.info("Testing initial SNN accuracy...")
    test_loss, test_acc = test_snn(model, test_loader, n_steps, criterion, device, logger)
    logger.info(f"Initial Accuracy: {test_acc:.2f}%")

    logger.info("Training SNN .... ")
    model = train_snn(train_loader, test_loader, model, n_steps, args.epochs, optimizer, scheduler, device, criterion, args, savename, logger)
    logger.info("ANN-to-SNN Conversion and Training Complete.")

    logger.info("Testing calibrated SNN accuracy...")
    test_loss, test_acc = test_snn(model, test_loader, n_steps, criterion, device, logger)
    logger.info(f"Final Accuracy: {test_acc:.2f}%")

    # Refresh memory
    del model
    torch.cuda.empty_cache()
    GlobalLogger.reset_logger()

