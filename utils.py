# utils.py

import torch
import torch.nn as nn

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total

# === TensorBoard Writer Utility ===

from torch.utils.tensorboard import SummaryWriter
import os

def get_writer(log_dir, experiment_name):
    path = os.path.join(log_dir, experiment_name)
    os.makedirs(path, exist_ok=True)
    writer = SummaryWriter(log_dir=path)
    return writer
