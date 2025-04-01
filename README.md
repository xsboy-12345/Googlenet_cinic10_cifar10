# GoogLeNet CINIC-10 → CIFAR-10 项目

## 项目目标

- 在 CINIC-10 上预训练 GoogLeNet
- 在 CIFAR-10 上微调预训练模型
- 从头训练 GoogLeNet 用作对比

## 使用说明

1. 下载 CINIC-10 数据集并解压到 `./CINIC-10`
2. 安装依赖：
```bash
pip install -r requirements.txt
```
3. 运行预训练：
```bash
python train.py
```
4. 微调训练：
```bash
python finetune.py
```
5. 从头训练：
```bash
python scratch.py
```