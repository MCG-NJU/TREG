import torch.nn as nn
import torch.nn.functional as F

class FPNUpBlock(nn.Module):
    def __init__(self, res_channels, planes=256, smooth_output=True, first_conv=False):
        super(FPNUpBlock, self).__init__()
        self.smooth_output = smooth_output
        self.first_conv = first_conv

        self.res_conv = nn.Conv2d(res_channels, planes, kernel_size=1, stride=1, padding=0)

        if self.smooth_output:
            self.pyramid_conv = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
            nn.init.kaiming_uniform_(self.pyramid_conv.weight, a=1)
            nn.init.constant_(self.pyramid_conv.bias, 0)


        nn.init.kaiming_uniform_(self.res_conv.weight, a=1)
        nn.init.constant_(self.res_conv.bias, 0)


    def forward(self, x, x_backbone):
        x_res = self.res_conv(x_backbone)
        if self.first_conv:
            return x_res
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        if self.smooth_output:
            out = self.pyramid_conv(x + x_res)
        else:
            out = x + x_res
        return out

class PyramidUpNet(nn.Module):
    def __init__(self, layer1_dim=1024, layer2_dim=512, layer3_dim=256, out_dim=256):
        super(PyramidUpNet, self).__init__()
        self.layer1 = FPNUpBlock(res_channels=layer1_dim, planes=out_dim, smooth_output=False, first_conv=True)
        self.up2 = FPNUpBlock(res_channels=layer2_dim, planes=out_dim, smooth_output=False, first_conv=False)
        self.up3 = FPNUpBlock(res_channels=layer3_dim, planes=out_dim, smooth_output=True, first_conv=False)

    def forward(self, backbone_feat):
        x = self.layer1(x=None, x_backbone=backbone_feat['layer3'])
        x = self.up2(x=x, x_backbone=backbone_feat['layer2'])
        x = self.up3(x=x, x_backbone=backbone_feat['layer1'])

        return x
