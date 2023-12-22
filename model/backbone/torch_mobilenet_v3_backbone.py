import torchvision

from mmengine.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class TorchMobileNetV3Backbone(BaseModule):
    def __init__(self, cfg):
        self.backbone = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights_backbone=True).backbone
    def forward(self, x):
        return self.backbone(x)["out"]