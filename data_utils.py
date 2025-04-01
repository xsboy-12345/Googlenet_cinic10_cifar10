# data_utils.py

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from configs import batch_size, cinic_path, cifar_path

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def get_cinic10_loader():
    trainset = datasets.ImageFolder(root=f"{cinic_path}/train", transform=transform)
    valset = datasets.ImageFolder(root=f"{cinic_path}/val", transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, valloader

def get_cifar10_loader(train=True):
    dataset = datasets.CIFAR10(root=cifar_path, train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2)