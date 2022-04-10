import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
# from base import Basemodel

class CNN(nn.Module):
    """
    Base Implement Model to training
    """
    def __init__(self, class_n, rate=0.1):
        super(CNN, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.classifier1 = nn.Linear(1000,25)
        self.dropout = nn.Dropout(rate)
    
    def forward(self, inputs):
        output = self.model(inputs)
        output = self.dropout(self.classifier1(output))# dropout 조심.
        return output