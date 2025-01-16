import numpy as np
from torch import nn
import torch
from tqdm import tqdm
from utils import *
import random
import os
import logging
import pickle
# Configure logging
logger = GlobalLogger.get_logger(__name__)

def seed_all(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    logger.info(f"Seeds set with seed={seed}")

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

def train_ann(train_dataloader, test_dataloader, model, epochs, device, loss_fn, lr=0.1, wd=5e-4, save=None, parallel=False, rank=0):
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
        tmp_acc, val_loss = eval_ann(test_dataloader, model, loss_fn, device, rank)
        if parallel:
            dist.all_reduce(tmp_acc)
        
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

