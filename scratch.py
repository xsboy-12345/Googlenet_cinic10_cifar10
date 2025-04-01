# scratch.py

import torch
import torch.nn as nn
import torch.optim as optim

from configs import *
from models.googlenet_bn import GoogLeNet
from data_utils import get_cifar10_loader
from utils import train, evaluate

trainloader = get_cifar10_loader(train=True)
testloader = get_cifar10_loader(train=False)
device = torch.device(device)

model = GoogLeNet(num_classes=num_classes)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    loss = train(model, trainloader, optimizer, criterion, device)
    acc = evaluate(model, testloader, device)
    print(f'[Epoch {epoch+1}] Loss: {loss:.4f}, Test Acc: {acc:.4f}')

# === TensorBoard Logging ===

from utils import get_writer

writer = get_writer("runs", "experiment")

# 在训练/评估循环中添加：
# writer.add_scalar('Loss/train', loss.item(), epoch)
# writer.add_scalar('Acc/val', acc, epoch)

# 在训练结束后关闭：
# writer.close()
