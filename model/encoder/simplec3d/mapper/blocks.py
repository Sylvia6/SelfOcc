import torch.nn.functional as F
import torch.nn as nn


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad, conv_3d_types="3D"):

    if conv_3d_types == "3D":

        return nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
            nn.BatchNorm3d(out_planes))

    elif conv_3d_types == "P3D":  # 3*3*3　to 1*3*3 + 3*1*1

        return nn.Sequential(

            nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride, padding=(0, 1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_planes, out_planes, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(out_planes))


class hourglass_PSMNet(nn.Module):
    def __init__(self, inplanes, conv_3d_types1, activate_fun):
        super(hourglass_PSMNet, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1, conv_3d_types = conv_3d_types1),
                                   activate_fun)

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1, conv_3d_types = conv_3d_types1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1, conv_3d_types = conv_3d_types1),
                                   activate_fun)

        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1, conv_3d_types = conv_3d_types1),
                                   activate_fun)

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(inplanes * 2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(inplanes))  # +x

        self.activate_fun = nn.Sequential(activate_fun)

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = self.activate_fun(pre + postsqu)
        else:
            pre = self.activate_fun(pre)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = self.activate_fun(self.conv5(out) + presqu)  # in:1/16 out:1/8
        else:
            post = self.activate_fun(self.conv5(out) + pre)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post

