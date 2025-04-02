import torch
import torch.nn as nn
import torch.optim as optim
import os

from configs import *
from models.googlenet_bn import GoogLeNet
from data_utils import get_cinic10_loader
from utils import train, evaluate, get_writer

# === 准备数据、模型、优化器 ===
trainloader, valloader = get_cinic10_loader()
device = torch.device(device)

model = GoogLeNet(num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# === 日志记录器（TensorBoard）===
writer = get_writer("runs", "googlenet_cinic10")

# === 模型保存路径 ===
best_acc = 0
patience = 3
trigger_times = 0
os.makedirs(model_path, exist_ok=True)

for epoch in range(epochs):
    loss = train(model, trainloader, optimizer, criterion, device)
    acc = evaluate(model, valloader, device)

    print(f'[Epoch {epoch+1}] Loss: {loss:.4f}, Val Acc: {acc:.4f}')

    # 写入 TensorBoard 日志
    writer.add_scalar('Loss/train', loss, epoch)
    writer.add_scalar('Acc/val', acc, epoch)

    # 模型保存 + early stopping 检查
    if acc > best_acc:
        best_acc = acc
        trigger_times = 0
        torch.save(model.state_dict(), f"{model_path}/googlenet_cinic10_best.pth")
        print("✅ New best model saved.")
    else:
        trigger_times += 1
        print(f"⚠️  No improvement. Trigger times: {trigger_times}")
        if trigger_times >= patience:
            print("🛑 Early stopping triggered.")
            break

# 最后完整模型也存一下（可选）
torch.save(model.state_dict(), f"{model_path}/googlenet_cinic10_last.pth")
writer.close()
