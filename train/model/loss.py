import torch
import torch.nn as nn
import torch.nn.functional as F

def nll_loss(output, target):
    return F.nll_loss(output, target)

def cross_entropy_loss(output, target, weight=None):
    return F.cross_entropy(output, target, weight=weight)

# TODO : LabelSmoothingLossì™€ FocalLoss naive Implement