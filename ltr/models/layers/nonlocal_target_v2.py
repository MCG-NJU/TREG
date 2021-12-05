import torch
from torch import nn
from torch.nn import functional as F

class NonLocalTarget(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True, topK=None, scale=None,
                 use_res=True):
        super(NonLocalTarget, self).__init__()

        self.topK = topK

        self.scale = scale

        self.sub_sample = sub_sample

        self.use_res = use_res

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

        # self.post_net = nn.Conv2d(4*6*6, 256, 3, 1, 1, bias=False)

        # if sub_sample:
        #     self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
        #     self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, targets_feat, test_feat, k=None, v=None):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        # init: (num_images, b, c, h, w)

        batch_size = targets_feat.size(1)

        t = targets_feat.permute(1, 2, 0, 3, 4) # (num_sequences, c, num_images, h, w)
        search = test_feat.permute(1, 2, 0, 3, 4)

        if v is not None:
            g_t = v
        else:
            g_t = self.g(t).view(batch_size, self.inter_channels, -1)
            g_t = g_t.permute(0, 2, 1)

        theta_s = self.theta(search).view(batch_size, self.inter_channels, -1)
        # theta_s = search.reshape(batch_size, self.inter_channels, -1)
        theta_s = theta_s.permute(0, 2, 1)

        if k is not None:
            phi_t = k
        else:
            phi_t = self.phi(t).view(batch_size, self.inter_channels, -1)
        # phi_t = self.phi(t).view(batch_size, self.inter_channels, -1)
        # phi_t = t.reshape(batch_size, self.inter_channels, -1)

        # whiten
        # theta_s_mean = theta_s.mean(1).unsqueeze(1)
        # phi_t_mean = phi_t.mean(2).unsqueeze(2)
        # theta_s -= theta_s_mean
        # phi_t -= phi_t_mean

        f = torch.matmul(theta_s, phi_t)  # ( , H*W, t*h*w)

        if self.topK is not None:
            mask_n = int((t.size(2) * t.size(3) * t.size(4)) - self.topK * (t.size(2) * t.size(3) * t.size(4)))
            sorted_inds = torch.argsort(f, dim=-1, descending=False)  # ( , H*W, t*h*w) 升序排序，将排名前(t*h*w-topK)的元素置0
            mask_inds = sorted_inds[..., :mask_n]  # ( , H*W, maskn)
            f = f.scatter_(dim=2, index=mask_inds, value=0.)
            norm_scale = 1.0 / (t.size(2) * t.size(3) * t.size(4) - mask_n)
            # norm_scale = 1.0 / (t.size(2) * t.size(3) * t.size(4))
            f_div_C = f * norm_scale
        else:
            # f_div_C = F.softmax(f, dim=-1)
            norm_scale = 1.0 / (t.size(2) * t.size(3) * t.size(4))
            f_div_C = f * norm_scale

        # f_div_C = F.softmax(f, dim=-1)
        # norm_scale = 1.0 / (t.size(2) * t.size(3) * t.size(4))
        # f_div_C = f * norm_scale

        y = torch.matmul(f_div_C, g_t) #(num_sequences, THW, C)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *search.size()[2:])
        if self.use_res is True:
            W_y = self.W(y) + search
        else:
            W_y = self.W(y)
        if self.scale is not None:
            W_y = W_y * self.scale
        # W_y = self.W(y)
        # W_y = y + search
        W_y = W_y.permute(2, 0, 1, 3, 4).contiguous()

        z = W_y.view(-1, *W_y.size()[2:])

        # f_corr = f.view(f.size(0), search.size(-2), search.size(-1), -1).permute(0, 3, 1, 2)
        # z = self.post_net(f_corr)


        return z, phi_t, g_t
