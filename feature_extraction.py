import torch
import torch.nn as nn
from spiking_layer_ours import *
from Models import modelpool
from Preprocess import datapool
import os
import argparse
from funcs import *
from utils import *
import numpy as np
import time
import sys
import calc_th_with_c as ft
import logging
from datetime import datetime

def extract_features(L=2):
    logger.info(f'Extracting features for layer L={L}, n_steps={n_steps}')
    logger.info("Loading dataset...")
    train_loader, test_loader = datapool(args.dataset, batch_size, 2, args.train_split)
    logger.info(f"Dataset loaded successfully. Training batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    with torch.no_grad():
        features = []
        Images = []
        count = 0
        
        try:
            if L == 1 and n_steps == 1:
                logger.info('Processing first layer features')
                for images, labels in train_loader:
                    Images.append(images)
                    outputs = model(images.to(device), L=L)
                    count += images.shape[0]
                    features.append(outputs.detach().cpu())
                    if count >= args.samples:
                        break
                
                Images = torch.cat(Images).numpy()
                np.save(f'{args.dataset}_Images.npy', Images)
                logger.info(f'Saved images array with shape {Images.shape}')
            else:
                logger.info('Loading saved images and processing features')
                Images = np.load(f'{args.dataset}_Images.npy')
                for i in range(Images.shape[0]//batch_size):
                    images = torch.Tensor(Images[i*batch_size:(i+1)*batch_size])
                    outputs = model(images.to(device), L=L)
                    count += images.shape[0]
                    features.append(outputs.detach().cpu())
                    if count >= args.samples:
                        break
            
            features = torch.cat(features).flatten().numpy()
            features = features[features > 0.0]
            features.sort()
            features = features.astype(np.longdouble)
            logger.info(f'Extracted {features.shape[0]} positive features')
            
            th_pos = ft.intl2(features, n_steps)
            for i in range(args.iter):
                th_pos = ft.optimize1(features, th_pos)
                error = ft.error(features, th_pos)
                logger.info(f'Iteration {i}: Error = {error}')
            
            thresholds_pos_all[L-1] = np.array(th_pos)
            thrs_in_list, thrs_out_list = ft.thrs_in_out(features, th_pos)
            thresholds_all[L-1, 0:n_steps] = np.array(thrs_out_list)
            thresholds_all[L-1, n_steps:] = np.array(thrs_in_list)
            
        except Exception as e:
            logger.error(f'Error in extract_features: {str(e)}', exc_info=True)
            raise


parser = argparse.ArgumentParser(description='PyTorch ANN-SNN Conversion')
parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset name',
                    choices=['cifar10', 'cifar100', 'imagenet', 'tiny-imagenet', 'fashion'])
parser.add_argument('--model', default='vgg16', type=str, help='Model name',
                    choices=['cifarnet', 'small', 'vgg16', 'resnet18', 'resnet20',
                             'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg16_normed',
                             'resnet18', 'resnet20', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
parser.add_argument('--checkpoint', default='./saved_models', type=str, help='Directory for saving models')
parser.add_argument('--iter', default=200, type=int, help='Number of iterations for finding th values')
parser.add_argument('--samples', default=10000, type=int, help='Number of iterations for finding th values')
parser.add_argument('--reference_models', default=4, type=int, help='Number of reference models')

args = parser.parse_args()

batch_size = 128
sample = 0
n_steps = 1

# Check device configuration and set accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

for model_idx in range(1, args.reference_models+1):
    # Configure logging
    GlobalLogger.initialize(
        os.path.join(
            primary_log_path, 
            f"model_{model_idx}", 
            f"threshold_i{args.iter}_N{str(args.samples)}.log"
        )
    )
    logger = GlobalLogger.get_logger(__name__)
    # Log initial configuration details
    logger.info("Starting SNN calibration")
    logger.info(f"Arguments: {args}")
    logger.info(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    # Load the specified model from the model pool
    logger.info(f"Loading model: {args.model} for dataset: {args.dataset}")
    savename = os.path.join(primary_model_path, f"model_{model_idx}", f"ann")
    model = modelpool(args.model, args.dataset)
    model.load_state_dict(torch.load(savename + '.pth'))
    model.to(device)
    # Define loss functoin
    criterion = nn.CrossEntropyLoss()
    num_relu = str(model).count('ReLU')
    logger.info(f'Model loaded successfully. Number of ReLU layers: {num_relu}')
    # Threshold calculation
    thresholds_all = np.zeros((num_relu, n_steps*2))
    thresholds_pos_all = np.zeros((num_relu, n_steps*2))
    for i in range(num_relu):
        logger.info(f'Processing ReLU layer {i+1}/{num_relu}')
        extract_features(L=i+1)
    # Save results
    np.save(f'{savename}_threshold_all_noaug{n_steps}.npy', thresholds_all)
    np.save(f'{savename}_threshold_pos_all_noaug{n_steps}.npy', thresholds_pos_all)
    logger.info('Successfully saved threshold arrays')

    GlobalLogger.reset_logger()