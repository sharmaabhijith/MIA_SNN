import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from spiking_layer_ours import *
from Models import modelpool
from Preprocess import datapool, get_dataloader_from_dataset, load_dataset, split_dataset
from torchvision.models.feature_extraction import create_feature_extractor
import os
import argparse
from funcs import *
import numpy as np
import time
import sys
import calc_th_with_c as ft
from copy import deepcopy
from utils import *


def isActivation(name):
    if 'relu' in name.lower():
        return True
    return False


def replace_activation_by_spike(model, thresholds, thresholds1, n_steps, counter=0):
    thresholds_new = deepcopy(thresholds)
    thresholds_new1 = deepcopy(thresholds1)

    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name], counter, thresholds_new = replace_activation_by_spike(module, thresholds_new, thresholds_new1, n_steps, counter)
        if isActivation(module.__class__.__name__.lower()):
            thresholds_new[counter, n_steps:] = thresholds_new1[counter, 1] / n_steps
            thresholds_new[counter, :n_steps] = thresholds_new1[counter, 0] / n_steps
            model._modules[name] = SPIKE_layer(thresholds_new[counter, n_steps:], thresholds_new[counter, 0:n_steps])
            counter += 1
    return model, counter, thresholds_new


def ann_to_snn(model, thresholds, thresholds1, n_steps):
    logger.info("Converting ANN to SNN...")
    model, counter, thresholds_new = replace_activation_by_spike(model, thresholds, thresholds1, n_steps)
    model = replace_maxpool2d_by_avgpool2d(model)
    model = replace_layer_by_tdlayer(model)
    logger.info("Conversion complete.")
    return model, thresholds_new


def test_snn(model, test_loader, n_steps, criterion, device):
    logger.info("Testing SNN...")
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        loss = 0
        for images, labels in test_loader:
            images = add_dimension(images, n_steps)
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images, L=0, t=n_steps)
            outputs = torch.sum(outputs, 1)
            _, predicted = torch.max(outputs.data/n_steps, 1)
            loss += criterion(outputs/n_steps, labels).item()*images.shape[0]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_loss = loss/total
        test_acc = 100 * correct / total
    logger.info(f"SNN Testing Complete. Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
    return test_loss, test_acc


def train_snn(train_dataloader, test_loader, model, n_steps, epochs, optimizer,
              scheduler, device, loss_fn, args, savename):
    logger.info("Starting SNN training...")
    model.to(device)
    best_epoch = 0
    best_acc = 0
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}...")
        model.train()
        epoch_loss = 0
        total = 0
        correct = 0

        for img, label in train_dataloader:
            img = add_dimension(img, n_steps)
            img = img.to(device)

            labels = label.to(device)
            outputs = model(img, L=0, t=n_steps) 
            outputs = torch.mean(outputs, 1)
            optimizer.zero_grad()
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()*img.shape[0]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        logger.info(f"Epoch {epoch+1} - Train Loss: {epoch_loss/total:.4f}, Train Accuracy: {100 * correct / total:.2f}%")
        scheduler.step()

        test_loss, test_acc = test_snn(model, test_loader, n_steps, loss_fn, device)
        if best_acc <= test_acc:
            save_path = f"{savename}_snn_T{n_steps}.pth"
            torch.save(model.state_dict(), save_path)
            best_acc = test_acc
            best_epoch = epoch
            logger.info(f"New Best Accuracy: {best_acc:.2f}% at Epoch {best_epoch+1}. Model saved to {save_path}.")

    return model


logger.info("Initializing ANN-to-SNN Conversion Script...")
parser = argparse.ArgumentParser(description='PyTorch ANN-SNN Conversion')
parser.add_argument('--t', default=300, type=int, help='T Latency length (Simulation time-steps)')
parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset name',
                    choices=['cifar10', 'cifar100', 'imagenet','tiny-imagenet','fashion'])
parser.add_argument('--model', default='vgg16', type=str, help='Model name',
                    choices=['vgg16', 'resnet18', 'resnet34', 'resnet50'])
parser.add_argument('--checkpoint', default='./saved_models', type=str, help='Directory for saving models')
parser.add_argument('--batchsize', default=64, type=int)
parser.add_argument('--lr', default=5e-3, type=float, help='Learning rate')
parser.add_argument('--wd', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--device', default='cuda:0', type=str)
parser.add_argument('--constant_lr', action='store_true')
parser.add_argument('--reference_models', default=4, type=int, help='Number of reference models')

args = parser.parse_args()

# Check device configuration and set accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_steps = args.t

# Creating directory to save trained models and their logs
primary_model_path = os.path.join(args.checkpoint, args.dataset, args.model, f"ref_models_{args.reference_models}")
primary_log_path = os.path.join("logs", args.dataset, args.model, f"ref_models_{args.reference_models}")

for model_idx in range(1, args.reference_models+1):
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
    if model_idx==1:
        # Load the dataset using the specified parameters
        logger.info("Loading dataset...")
        dataset = load_dataset(args.dataset, logger)
        try:
            with open(os.path.join(primary_log_path,"data_splits.pkl"), 'rb') as file:
                data_split_info = pickle.load(file)
        except FileNotFoundError:
            logging.info(f"data_splits.pkl not found in: {primary_log_path}")
    # Creating dataloader
    train_idxs = data_split_info[model_idx]["train"]
    test_idxs = data_split_info[model_idx]["test"]
    logger.info(
        f"Training model {model_idx}: Train size {len(train_idxs)}, Test size {len(test_idxs)}"
    )
    logger.info("Creating dataloader...")
    train_loader = get_dataloader(Subset(dataset, train_idxs), batch_size=batch_size, shuffle=True)
    test_loader = get_dataloader(Subset(dataset, test_idxs), batch_size=batch_size)
    logger.info(
        f"Dataset loaded successfully. Training batches: {len(train_loader)}, 
        Test batches: {len(test_loader)}"
    )
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
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    para1, para2, para3 = regular_set(model)
    optimizer = torch.optim.SGD([
                                {'params': para1, 'weight_decay': args.wd},
                                {'params': para2, 'weight_decay': args.wd},
                                {'params': para3, 'weight_decay': args.wd}
                                ],
                                lr=args.lr, momentum=0.9)
    if args.constant_lr:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, verbose=True)

    logger.info("Testing initial SNN accuracy...")
    test_loss, test_acc = test_snn(model, test_loader, n_steps, criterion, device)
    logger.info(f"Initial Accuracy: {test_acc:.2f}%")

    logger.info("Training SNN .... ")
    model = train_snn(train_loader, test_loader, model, n_steps, args.epochs, optimizer, scheduler, device, criterion, args, savename)
    logger.info("ANN-to-SNN Conversion and Training Complete.")

    logger.info("Testing calibrated SNN accuracy...")
    test_loss, test_acc = test_snn(model, test_loader, n_steps, criterion, device)
    logger.info(f"Final Accuracy: {test_acc:.2f}%")
