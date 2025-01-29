import os
import pickle
import torch
import random
import logging
import numpy as np
import torch.nn as nn
from copy import deepcopy
from typing import Optional
from spiking_layer_ours import *
from torch.nn.parameter import Parameter
from torch.nn import functional as F

def seed_all(logger, seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    logger.info(f"Seeds set with seed={seed}")

class GlobalLogger:
    _initialized = False
    _log_file = None

    @classmethod
    def initialize(cls, log_file):
        if not cls._initialized:
            cls._log_file = log_file
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
            for handler in handlers:
                handler.setFormatter(formatter)
                root_logger.addHandler(handler)
            
            cls._initialized = True
    
    @classmethod
    def reset_logger(cls):
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:  # Copy the list to avoid iteration issues
            root_logger.removeHandler(handler)
        cls._initialized = False

    @classmethod
    def get_logger(cls, name=None):
        return logging.getLogger(name)

def regular_set(model, paras=([],[],[])):
    for n, module in model._modules.items():
       
        if 'batchnorm' in module.__class__.__name__.lower():
            for name, para in module.named_parameters():
                paras[2].append(para)
                #print("paras[2]")
        elif len(list(module.children())) > 0:
            paras = regular_set(module, paras)
            #print("recursive")
        elif module.parameters() is not None:
            for name, para in module.named_parameters():
                paras[1].append(para)
                #print("paras[1]")
    return paras

class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly https://github.com/fangwei123456/spikingjelly
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        #print(x_seq.flatten(0, 1).contiguous().shape)
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)

class tdLayer(nn.Module):
    def __init__(self, layer):
        super(tdLayer, self).__init__()
        self.layer = SeqToANNContainer(layer)
       

    def forward(self, x):
        x_ = self.layer(x)
      
        return x_

class tdBatchNorm(nn.Module):
    def __init__(self, out_panel):
        super(tdBatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(out_panel)
        self.seqbn = SeqToANNContainer(self.bn)

    def forward(self, x):
        y = self.seqbn(x)
        return y


def replace_layer_by_tdlayer(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_layer_by_tdlayer(module)
        if module.__class__.__name__ == 'Conv2d':
            model._modules[name] = tdLayer(model._modules[name])
        if module.__class__.__name__ == 'Linear':
            model._modules[name] = tdLayer(model._modules[name])
        if module.__class__.__name__ == 'BatchNorm2d':
            model._modules[name] = tdLayer(model._modules[name])
        if module.__class__.__name__ == 'AvgPool2d':
            model._modules[name] = tdLayer(model._modules[name])
        if module.__class__.__name__ == 'Flatten':
            model._modules[name] = nn.Flatten(start_dim=-3,end_dim=-1)
        if module.__class__.__name__ == 'Dropout':
            model._modules[name] = tdLayer(model._modules[name])
        if module.__class__.__name__ == 'AdaptiveAvgPool2d':
            model._modules[name] = tdLayer(model._modules[name])       
    return model

def isActivation(name):
    if 'spike_layer' in name.lower() :
        return True
    return False

def replace_maxpool2d_by_avgpool2d(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_maxpool2d_by_avgpool2d(module)
        if module.__class__.__name__ == 'MaxPool2d':
            model._modules[name] = nn.AvgPool2d(kernel_size=module.kernel_size,
                                                stride=module.stride,
                                                padding=module.padding)
    return model


def add_dimension(x, T):
    x.unsqueeze_(1)
    x = x.repeat(1, T, 1, 1, 1)
    return x

def isActivation_spike(name):
        if 'spike_layer' in name.lower():
            return True
        return False

def snn_to_ann(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = snn_to_ann(module)
        if module.__class__.__name__ == 'SPIKE_layer':
            model._modules[name] = nn.Linear(module.in_features, module.out_features)
            model._modules[name].weight = Parameter(module.weight)
            model._modules[name].bias = Parameter(module.bias)
        elif module.__class__.__name__ == 'tdLayer':
            model._modules[name] = module.layer.module
        elif module.__class__.__name__ == 'Flatten':
            model._modules[name] = nn.Flatten()
    return model

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


def ann_to_snn(model, thresholds, thresholds1, n_steps, logger):
    logger.info("Converting ANN to SNN...")
    model, counter, thresholds_new = replace_activation_by_spike(model, thresholds, thresholds1, n_steps)
    model = replace_maxpool2d_by_avgpool2d(model)
    model = replace_layer_by_tdlayer(model)
    logger.info("Conversion complete.")
    return model, thresholds_new


def test_snn(model, test_loader, n_steps, criterion, device, logger):
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
              scheduler, device, loss_fn, args, savename, logger):
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
            save_path = f"{savename}_snn_T{n_steps}"
            torch.save(model.state_dict(), save_path + ".pth")
            with open(save_path + ".pkl", "wb") as f:
                pickle.dump(model.state_dict(), f)
            best_acc = test_acc
            best_epoch = epoch
            logger.info(f"New Best Accuracy: {best_acc:.2f}% at Epoch {best_epoch+1}. Model saved to {save_path}.")

    return model

def test_ann(test_dataloader, model, loss_fn, device, logger, rank=0):
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

def train_ann(train_dataloader, test_dataloader, model, epochs, device, loss_fn, logger, lr=0.1, wd=5e-4, save=None, rank=0):
    os.makedirs(os.path.dirname(save), exist_ok=True)
    logger.info("Starting training...")
    logger.info(f"Parameters: epochs={epochs}, learning_rate={lr}, weight_decay={wd}")

    model.cuda(device)
    para1, para2, para3 = regular_set(model)
    
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	#optimizer = torch.optim.SGD
    #(
    #        [
    #            {'params': para1, 'weight_decay': wd}, 
    #            {'params': para2, 'weight_decay': wd}, 
    #            {'params': para3, 'weight_decay': wd}
    #        ],
    #        lr=lr, 
    #        momentum=0.1
	#)
	
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0
    for epoch in range(epochs):
        total = torch.tensor(0.).cuda(device)
        epoch_loss = 0
        length = 0
        model.train()
        logger.info(f"Epoch {epoch + 1}/{epochs} starting...")
        for img, label in tqdm(train_dataloader, desc=f"Epoch {epoch + 1} Progress", leave=False):
            img = img.cuda(device)
            label = label.cuda(device)
            optimizer.zero_grad()
            out = model(img)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            length += len(label)
            total += (label==out.max(1)[1]).sum().data

        train_acc = total / length
        train_loss = epoch_loss / length
        tmp_acc, val_loss = test_ann(test_dataloader, model, loss_fn, device, logger, rank)
        
        logger.info(f"Epoch {epoch + 1} -> Training Accuracy: {train_acc:.4f}, Validation Accuracy: {tmp_acc:.4f}")

        if tmp_acc >= best_acc:
            torch.save(model.state_dict(), save + '.pth')
            with open(save + ".pkl", "wb") as f:
                pickle.dump(model.state_dict(), f)
            best_acc = tmp_acc
            logger.info(f"New best model saved with accuracy: {best_acc:.4f}")

        scheduler.step()
    
    logger.info(f"Training completed. Best accuracy: {best_acc:.4f}")
    return best_acc, model




