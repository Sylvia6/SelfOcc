import torch
from ..utils import activations

from mmengine.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class CompressNeck(BaseModule):
    def __init__(self, inc, outc, activation):
        super(CompressNeck, self).__init__()
        self.conv = torch.nn.Conv2d(inc, outc, kernel_size=1, stride=1, padding=0)
        self.bn = torch.nn.BatchNorm2d(outc)
        self.activation = activations["activation"](inplace=True)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))