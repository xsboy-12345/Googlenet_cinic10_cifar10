# models/googlenet_bn.py
import torch
import torch.nn as nn
from torchvision.models import googlenet

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()
        self.model = googlenet(pretrained=False, aux_logits=False)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
