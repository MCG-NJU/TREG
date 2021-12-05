import torch
from torch import nn
from torch.nn import functional as F
import math

def xcorr_depthwise(x, kernel):
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out

class CorrAtt(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(CorrAtt, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv3d(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv3d(in_channels=self.inter_channels, out_channels=self.in_channels,
                               kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.spatial_att = nn.Sequential(nn.Conv2d(5*5, 1, 5, padding=2, bias=False),
                                         # nn.BatchNorm2d(1),
                                         nn.Sigmoid())

        # self.post_net = nn.Conv2d(5*5, 256, 3, 1, 1, bias=False)
        # for m in self.post_net.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #         if m.bias is not None:
        #             m.bias.data.zero_()
            # nn.GroupNorm(32, 256),
            # nn.ReLU(),

        # channel attention
        # self.conv_kernel = nn.Sequential(
        #     nn.Conv2d(in_channels, inter_channels, kernel_size=3),
        #     nn.ReLU(inplace=True),
        # )
        # self.conv_search = nn.Sequential(
        #     nn.Conv2d(in_channels, inter_channels, kernel_size=3),
        #     nn.ReLU(inplace=True),
        # )
        #
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #         if m.bias is not None:
        #             m.bias.data.zero_()


        # if sub_sample:
        #     self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
        #     self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, targets_feat, test_feat):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        # init: (num_images, b, c, h, w)
        batch_size = targets_feat.size(1)

        t = targets_feat.permute(1, 2, 0, 3, 4).contiguous() # (num_sequences, c, num_images, h, w)
        # print("t shape: {}".format(t.shape))
        search = test_feat.permute(1, 2, 0, 3, 4).contiguous()
        # print("search shape: {}".format(search.shape))

        s = search.view(batch_size, self.inter_channels, -1)
        s = s.permute(0, 2, 1)

        t = t.view(batch_size, self.inter_channels, -1)

        f = torch.matmul(s, t)  # ( , T*H*W, h*w)

        f_corr = f.view(batch_size, test_feat.size(0), test_feat.size(-2), test_feat.size(-1), -1).permute(1, 0, 4, 2, 3).contiguous()
        f_corr = f_corr.view(test_feat.size(0) * batch_size, *f_corr.size()[2:])
        # print("f_c shape: {}".format(f_corr.shape))
        spatial_att = self.spatial_att(f_corr)

        search = test_feat.view(test_feat.size(0) * batch_size, *test_feat.size()[2:])
        spa_aug_feat = search * spatial_att
        # z = self.post_net(f_corr)


        # # channel attention
        # tH, tW = targets_feat.shape[-2:]
        # num_test_image, batch_size, C, H, W = test_feat.shape
        # targets_feat = targets_feat.repeat(num_test_image, 1, 1, 1, 1)
        # targets_feat = targets_feat.reshape(-1, C, tH, tW)
        # test_feat = test_feat.reshape(-1, C, H, W)
        #
        # targets_feat = self.conv_kernel(targets_feat)
        # search_feat = self.conv_search(test_feat)
        #
        # test_padding_size = tH // 2
        # pad_tuple = (test_padding_size, test_padding_size, test_padding_size, test_padding_size)
        # search_feat = F.pad(search_feat, pad=pad_tuple, mode='replicate')
        #
        # feature = xcorr_depthwise(search_feat, targets_feat)
        #
        # F.interpolate(x_36, size=(72, 72))

        return spa_aug_feat