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

parser = argparse.ArgumentParser(description='PyTorch ANN-SNN Conversion')
parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset name',
                    choices=['cifar10', 'cifar100', 'imagenet', 'tiny-imagenet', 'fashion'])
parser.add_argument('--train_split', default=-1, type=float, help='Train Test Dataset Split')
parser.add_argument('--model', default='vgg16', type=str, help='Model name',
                    choices=['cifarnet', 'small', 'vgg16', 'resnet18', 'resnet20',
                             'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg16_normed',
                             'resnet18', 'resnet20', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
parser.add_argument('--checkpoint', default='./saved_models', type=str, help='Directory for saving models')
parser.add_argument('--iter', default=200, type=int, help='Number of iterations for finding th values')
parser.add_argument('--samples', default=10000, type=int, help='Number of iterations for finding th values')
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
GlobalLogger.initialize(f"{exp_logs_path}/threshold_{args.model}_{args.dataset}_i{args.iter}_N{str(args.samples)}.log")
logger = GlobalLogger.get_logger(__name__)


args.mid = f'{args.dataset}_{args.model}'
savename = os.path.join(exp_models_path, args.mid)+"_new"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')
batch_size = 128

sample = 0
n_steps = 1
thresholds_13 = np.zeros(n_steps)

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

try:
    model = modelpool(args.model, args.dataset)
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(savename + '.pth'))
    model.to(device)
    
    num_relu = str(model).count('ReLU')
    logger.info(f'Model loaded successfully. Number of ReLU layers: {num_relu}')
    
    thresholds_all = np.zeros((num_relu, n_steps*2))
    thresholds_pos_all = np.zeros((num_relu, n_steps*2))
    
    for i in range(num_relu):
        logger.info(f'Processing ReLU layer {i+1}/{num_relu}')
        extract_features(L=i+1)
    
    # Save results
    np.save(f'{savename}_threshold_all_noaug{n_steps}.npy', thresholds_all)
    np.save(f'{savename}_threshold_pos_all_noaug{n_steps}.npy', thresholds_pos_all)
    logger.info('Successfully saved threshold arrays')
    
except Exception as e:
    logger.error(f'Fatal error in main execution: {str(e)}', exc_info=True)
    raise
