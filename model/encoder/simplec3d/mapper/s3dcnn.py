import math
import torch.nn.functional as F
import torch.nn as nn

from mmengine.registry import MODELS
from mmengine.model import BaseModule
from .blocks import *

from ....utils import activations, IdentityLayer



@MODELS.register_module()
class ConvBnActivation3D(BaseModule):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 activation="relu"):
        super(ConvBnActivation3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.noliner = activations[activation]

    def forward(self, x):
        return self.noliner(self.bn(self.conv(x)))



@MODELS.register_module()
class Residual3DCNN(BaseModule):
    def __init__(self, in_channels, out_channels, planes, layers=3, kernel_size=3, stride=1, pad=1, conv_3d_types='3D', activation="relu"):
        super(Residual3DCNN, self).__init__()
        assert layers >= 2
        self.net_layers = nn.ModuleList()
        channels = [in_channels, ] + [planes, ] * (layers - 1) + [out_channels, ]
        for i in range(layers):
            inc = channels[i]
            outc = channels[i + 1]
            block = nn.ModuleList()
            block.append(self.get_by_pass(inc, outc, conv_3d_types))
            block.append(convbn_3d(planes,  planes, kernel_size=kernel_size, stride=stride, pad=pad, conv_3d_types = conv_3d_types))
            block.append(activations[activation])
            self.net_layers.append(block)

    def get_by_pass(self, inc, outc, conv_3d_types='3D'):
        if inc == outc:
            return IdentityLayer()
        return convbn_3d(inc, outc, kernel_size=1, stride=1, pad=0, conv_3d_types = conv_3d_types)

    def forward(self, x):
        for i in range(len(self.net_layers)):
            by_pass, net, activation_fun = self.blocks[i]
            x = activation_fun(by_pass(x) + net(x))
        return x


@MODELS.register_module()
class S3DCNN(BaseModule):
    def __init__(self,  input_planes = 64, out_planes = 1, planes = 16,  conv_3d_types1 = "3D", activation= 'relu'):
        super(S3DCNN, self).__init__()
        self.out_planes = out_planes
        activate_fun = activations[activation]

        self.dres0 = nn.Sequential(convbn_3d(input_planes, planes*2, 3, 1, 1, conv_3d_types =  conv_3d_types1),
                                     activate_fun,
                                     convbn_3d(planes*2, planes*2, 3, 1, 1, conv_3d_types =  conv_3d_types1),
                                     activate_fun)


        self.dres1 = nn.Sequential(convbn_3d(planes*2, planes*2, 3, 1, 1, conv_3d_types = conv_3d_types1),
                                   activate_fun,
                                   convbn_3d(planes*2, planes*2, 3, 1, 1, conv_3d_types = conv_3d_types1))

        self.dres2 = hourglass_PSMNet(planes*2, conv_3d_types1 = conv_3d_types1, activate_fun = activate_fun)

        self.dres3 = hourglass_PSMNet(planes*2, conv_3d_types1 = conv_3d_types1, activate_fun = activate_fun)

        self.dres4 = hourglass_PSMNet(planes*2, conv_3d_types1 = conv_3d_types1, activate_fun = activate_fun)


        self.classif1 = nn.Sequential(convbn_3d(planes*2, planes*2, 3, 1, 1, conv_3d_types =  conv_3d_types1),
                                      activate_fun,
                                      nn.Conv3d(planes*2, out_planes, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif2 = nn.Sequential(convbn_3d(planes*2, planes*2, 3, 1, 1, conv_3d_types =  conv_3d_types1),
                                      activate_fun,
                                      nn.Conv3d(planes*2, out_planes, kernel_size=3, padding=1, stride=1,bias=False))



        self.classif3 = nn.Sequential(convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1),
                                      activate_fun,

                                      convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1),
                                      activate_fun,

                                      convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1),
                                      activate_fun,

                                      convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1),
                                      activate_fun,

                                      convbn_3d(planes * 2, self.out_planes, 3, 1, 1, conv_3d_types=conv_3d_types1),)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, cost):

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)

        out1 = out1+cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2+cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)

        cost3 = self.classif3(out3)

        return [cost3]


def pad_to_shape(this, shp):
    """
    Pads this image with zeroes to shp.
    Args:
        this: image tensor to pad
        shp: desired output shape
    Returns:
        Zero-padded tensor of shape shp.
    """
    if len(shp) == 4:
        pad = (0, shp[3] - this.shape[3], 0, shp[2] - this.shape[2])
    elif len(shp) == 5:
        pad = (0, shp[4] - this.shape[4], 0, shp[3] - this.shape[3], 0, shp[2] - this.shape[2])
    return F.pad(this, pad)



if __name__ == '__main__':
    pass
