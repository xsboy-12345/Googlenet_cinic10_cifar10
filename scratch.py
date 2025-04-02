import torch
import torch.nn as nn
import torch.optim as optim
import os

from configs import *
from models.googlenet_bn import GoogLeNet
from data_utils import get_cifar10_loader
from utils import train, evaluate, get_writer

# === 准备数据和设备 ===
trainloader = get_cifar10_loader(train=True)
testloader = get_cifar10_loader(train=False)
device = torch.device(device)

# === 模型和优化器 ===
model = GoogLeNet(num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# === 日志和模型保存 ===
writer = get_writer("runs", "googlenet_scratch")
os.makedirs(model_path, exist_ok=True)
best_acc = 0
trigger_times = 0
patience = 3

# === 训练循环 ===
for epoch in range(epochs):
    loss = train(model, trainloader, optimizer, criterion, device)
    acc = evaluate(model, testloader, device)

    print(f'[Epoch {epoch+1}] Loss: {loss:.4f}, Test Acc: {acc:.4f}')

    # TensorBoard 日志
    writer.add_scalar('Loss/train', loss, epoch)
    writer.add_scalar('Acc/test', acc, epoch)

    # 模型保存 & Early Stopping
    if acc > best_acc:
        best_acc = acc
        trigger_times = 0
        torch.save(model.state_dict(), f"{model_path}/googlenet_scratch_best.pth")
        print("✅ New best model saved.")
    else:
        trigger_times += 1
        print(f"⚠️  No improvement. Trigger times: {trigger_times}")
        if trigger_times >= patience:
            print("🛑 Early stopping triggered.")
            break

# === 可选：保存最后模型
torch.save(model.state_dict(), f"{model_path}/googlenet_scratch_last.pth")
writer.close()
