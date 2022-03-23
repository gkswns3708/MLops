import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class CNN(nn.Module):
    def __init__(self, class_n, rate=0.1):
        super(CNN, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.classifier1 = nn.Linear(1000, 25)
        self.dropout = nn.Dropout(rate)

    def forward(self, inputs):
        output = self.model(inputs)
        output = self.dropout(self.classifier1(output))
        return output
