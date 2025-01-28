import os
import torch
from tqdm import tqdm
import torch.nn as nn
from torchvision.models import vgg16, resnet18

def select_model(model_name, n_classes):
    if model_name=="vgg16":
        model = vgg16(pretrained=True)
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, n_classes) 
    elif model_name=="resnet18":
        model = resnet18(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, n_classes)
    else:
        model = None
    return model

def eval_model(trained_model, test_loader, criterion, device):
    trained_model.eval()
    correct = 0
    total = 0
    overall_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = trained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            overall_loss += loss.item()  
    accuracy = correct / total
    avg_loss = overall_loss / total
    return accuracy, avg_loss

def train_model(model, train_loader, test_loader, optimizer, device, modelpath, n_epochs):
    criterion = nn.CrossEntropyLoss() 
    best_acc = 0
    for epoch in range(n_epochs):
        running_loss = 0.0
        total=0
        correct = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} Progress", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
        train_acc = correct/total
        val_acc, val_loss = eval_model(model, test_loader, criterion, device)        
        print(f"Epoch {epoch + 1} -> Training Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")
        if val_acc >= best_acc:
            torch.save(model.state_dict(), modelpath)
            best_acc = val_acc
            print(f"New best model saved with accuracy: {best_acc:.4f}")
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {running_loss/len(train_loader):.4f}")






