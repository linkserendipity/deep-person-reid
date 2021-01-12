from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from IPython import embed

class ResNet50(nn.Module):
    def __init__(self, num_classes, loss={'softmax, metric'}, **kwargs):
            super(ResNet50, self).__init__()
            resnet50 = torchvision.models.resnet50(pretrained=True)
            # resnet50 = torchvision.models.resnet50(pretraind=True)
            embed()

if __name__ == "__main__":
    model = ResNet50(num_classes = 751)    