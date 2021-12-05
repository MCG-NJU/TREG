import torch
from torch import nn
from torch.nn import functional as F
import math

class PixelCorr(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(PixelCorr, self).__init__()

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


        self.post_net = nn.Conv2d(5*5, 256, 3, 1, 1, bias=False)
        for m in self.post_net.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            # nn.GroupNorm(32, 256),
            # nn.ReLU(),


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

        # g_t = self.g(t).view(batch_size, self.inter_channels, -1)
        # g_t = t.permute(0, 2, 1)

        s = search.view(batch_size, self.inter_channels, -1)
        s = s.permute(0, 2, 1)

        t = t.view(batch_size, self.inter_channels, -1)

        f = torch.matmul(s, t)  # ( , T*H*W, h*w)
        # print("f shape: {}".format(f.shape))
        # f_div_C = F.softmax(f, dim=-1)
        # norm_scale = 1.0 / (t.size(2) * t.size(3) * t.size(4))
        # f_div_C = f * norm_scale
        #
        # y = torch.matmul(f_div_C, g_t) #(num_sequences, THW, C)
        # y = y.permute(0, 2, 1).contiguous()
        # y = y.view(batch_size, self.inter_channels, *search.size()[2:])
        # W_y = self.W(y) + search
        # W_y = W_y.permute(2, 0, 1, 3, 4).contiguous()
        #
        # z = W_y.view(-1, *W_y.size()[2:])

        f_corr = f.view(batch_size, test_feat.size(0), test_feat.size(-2), test_feat.size(-1), -1).permute(1, 0, 4, 2, 3).contiguous()
        f_corr = f_corr.view(test_feat.size(0) * batch_size, *f_corr.size()[2:])

        # f_corr = f.view(batch_size * test_feat.size(0), test_feat.size(-2), test_feat.size(-1), -1).permute(0, 3, 1, 2)
        # print("f_c shape: {}".format(f_corr.shape))
        z = self.post_net(f_corr)

        return z