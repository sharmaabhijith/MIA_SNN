import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from spiking_layer_ours import *
from Models import modelpool
from Preprocess import datapool
from torchvision.models.feature_extraction import create_feature_extractor
import os
import argparse
from funcs import *
import numpy as np
import time
import logging
from utils import *

parser = argparse.ArgumentParser(description='PyTorch ANN-SNN Conversion')

parser.add_argument('--t', default=300, type=int, help='T Latency length (Simulation time-steps)')
parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset name',
                    choices=['cifar10', 'cifar100', 'tiny-imagenet', 'imagenet', 'fashion'])
parser.add_argument('--train_split', default=0.9, type=float, help='Train Test Dataset Split')
parser.add_argument('--model', default='vgg16', type=str, help='Model name',
                    choices=['small', 'vgg16', 'resnet18', 'resnet20', 'vgg16_no_bn',
                             'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg16_normed', 'alexnet',
                             'resnet18', 'resnet19', 'resnet20', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'cifarnet'])
parser.add_argument('--checkpoint', default='./saved_models', type=str, help='Directory for saving models')
parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
parser.add_argument('--wd', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--exp_type', default='RMIA', type=str, help='Model name',
                    choices=['ANN2SNN', 'RMIA', 'RMIA_SNN'])


args = parser.parse_args()

# Creating directory to save trained models
exp_models_path = os.path.join(args.checkpoint, args.exp_type, args.dataset, args.model)
if os.path.exists(exp_models_path) is False:
    print("Creating model directory:", exp_models_path)
    os.makedirs(exp_models_path)

# Creating directory to save Log files
exp_logs_path = os.path.join("logs", args.exp_type, args.dataset, args.model)
if os.path.exists(exp_logs_path) is False:
    print("Creating model directory:", exp_logs_path)
    os.makedirs(exp_logs_path)
    
# Configure logging
GlobalLogger.initialize(f"{exp_logs_path}/ann_train_{args.model}_{args.dataset}_lr{args.lr}.log")
logger = GlobalLogger.get_logger(__name__)

args.mid = f'{args.dataset}_{args.model}'
savename = os.path.join(exp_models_path, args.mid) + "_new"

# Log initial configuration
logger.info("Starting ANN-SNN Conversion")
logger.info(f"Arguments: {args}")
logger.info(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
logger.info(f"Checkpoint path: {savename}")

# Check Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Hyper-parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = args.epochs
batch_size = 256

try:
    # Load model
    logger.info(f"Loading model: {args.model} for dataset: {args.dataset}")
    model = modelpool(args.model, args.dataset)
    model.to(device)

    # Load data
    logger.info("Loading dataset...")
    train_loader, test_loader = datapool(args.dataset, batch_size, 2, args.train_split, shuffle=True)
    logger.info(f"Dataset loaded successfully. Training batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()

    # Train ANN model
    logger.info("Starting training...")
    train_ann(train_loader, test_loader, model, num_epochs, device, criterion, args.lr, args.wd, savename)
    logger.info("Training completed successfully.")

except Exception as e:
    logger.error("An error occurred during execution.", exc_info=True)
    raise

