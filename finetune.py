# finetune.py

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
model.load_state_dict(torch.load(f"{model_path}/googlenet_cinic10.pth"))
model = model.to(device)

for name, param in model.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
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
