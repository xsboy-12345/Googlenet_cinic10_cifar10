import torch
import torch.nn as nn
import torch.optim as optim
import os

from configs import *
from models.googlenet_bn import GoogLeNet
from data_utils import get_cifar10_loader
from utils import train, evaluate, get_writer

# === 数据加载 ===
trainloader = get_cifar10_loader(train=True)
testloader = get_cifar10_loader(train=False)
device = torch.device(device)

# === 加载预训练模型 ===
model = GoogLeNet(num_classes=num_classes)
pretrained_path = os.path.join(model_path, "googlenet_cinic10_best.pth")
if not os.path.exists(pretrained_path):
    raise FileNotFoundError(f"❌ 找不到预训练模型: {pretrained_path}")
model.load_state_dict(torch.load(pretrained_path, map_location=device))
model = model.to(device)

# === 冻结除了 fc 的层 ===
for name, param in model.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False

# === 优化器、损失函数、日志工具 ===
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
writer = get_writer("runs", "googlenet_finetune")
os.makedirs(model_path, exist_ok=True)

# === Early Stopping 设置 ===
best_acc = 0
trigger_times = 0
patience = 3

# === 训练循环 ===
for epoch in range(epochs):
    loss = train(model, trainloader, optimizer, criterion, device)
    acc = evaluate(model, testloader, device)
    print(f'[Epoch {epoch+1}] Loss: {loss:.4f}, Test Acc: {acc:.4f}')

    # === 写入 TensorBoard ===
    writer.add_scalar('Loss/train', loss, epoch)
    writer.add_scalar('Acc/test', acc, epoch)

    # === 保存最佳模型
    if acc > best_acc:
        best_acc = acc
        trigger_times = 0
        torch.save(model.state_dict(), os.path.join(model_path, "googlenet_finetuned_best.pth"))
        print("✅ New best model saved.")
    else:
        trigger_times += 1
        print(f"⚠️  No improvement. Trigger times: {trigger_times}")
        if trigger_times >= patience:
            print("🛑 Early stopping triggered.")
            break

# === 可选：保存最后模型
torch.save(model.state_dict(), os.path.join(model_path, "googlenet_finetuned_last.pth"))
writer.close()
