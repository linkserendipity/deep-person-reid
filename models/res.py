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
            self.base = nn.Sequential(*list(resnet50.children())[:-2]) #
            self.classifier = nn.Linear(2048, num_classes)
            
    def forward(self, x):
        x =self.base(x)      
        x = F.avg_pool2d(x, x.size()[:2])
        f = x.view(x.size(0), -1)
        # normalize
        # f = 1.*f / (torch.norm(f, 2, dim=-1, keepdim=True).expand_as(f) + 1e-12)

        if not self.training:
            return f
        y = classifier(f)
        return y
        

if __name__ == "__main__":
    model = ResNet50(num_classes = 751)    
    imgs = torch.Tensor(32, 3, 256, 128)
    f = model(imgs)
    embed()