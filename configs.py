import torch

batch_size = 128
epochs = 20
learning_rate = 0.001
num_classes = 10

# 数据路径
cinic_path = './CINIC-10'
cifar_path = './data'

# 模型保存路径
model_path = './checkpoints'

# 自动根据是否有GPU选择设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
