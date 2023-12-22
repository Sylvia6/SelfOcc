import torch

from mmengine.registry import MODELS
from mmengine.model import BaseModule

from ....utils import activations

@MODELS.register_module()
class DeconvUpsampleNeck3d(BaseModule):
    def __init__(self, inc, activation, output_mapper, upsample_number=3):
        super(DeconvUpsampleNeck3d, self).__init__()
        self.upsample_layers = torch.nn.ModuleList()
        for i in range(upsample_number):
            self.upsample_layers.append(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose3d(inc, inc, kernel_size=2, stride=2, padding=0),
                    torch.nn.BatchNorm3d(inc),
                    activations[activation]))
        self.output_mapper = MODELS.build(**output_mapper)

    def forward(self, x):
        for i in range(len(self.upsample_layers)):
            x = self.upsample_layers[i](x)
        return self.output_mapper(x)