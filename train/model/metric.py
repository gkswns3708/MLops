from numpy.lib.function_base import average
import torch
from torchmetrics.functional import f1 as _f1

def accuarcy(output, target):
    # TODO : 이거 왜 torch.no_grad 썼는지 이해하기.
    with torch.no_grad():
        # TODO : 이거 왜 dim 1인지 이해하기. 아마 이런걸 잘 생각해야 좋은 ML engineer일 듯 함.
        # 당연하게도 predicition logit들 중에서 가장 큰 값을 prediction 값으로 생각해야함
        # 2차원 행렬이므로 1개의 행에서 가장 큰 열을 찾아야 함 그래서 dim = 1이라고 예측 됨.
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred==target).detach()
    return correct / len(target)

def f1(output, target, num_classes=25):
    return _f1(output, target, num_classes=num_classes, average='macro')