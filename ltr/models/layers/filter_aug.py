import torch
from torch import nn
from torch.nn import functional as F

class AugFilter(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(AugFilter, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=3, stride=1, padding=1)


        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)


        # self.post_net = nn.Conv2d(4*6*6, 256, 3, 1, 1, bias=False)

        # if sub_sample:
        #     self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
        #     self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, init_filter, targets_feat, bb, feat_stride=4):
        '''
        :param init_filter: (b, c, h, w)
               targets_feat: (num_images, b, c, H, W)
               mask: (num_images, b, H, W), target区域的值为0，其他为1
               bb: Target boxes (x,y,w,h) for the train samples in image coordinates. Dims (images, sequences, 4).
        :return:
        '''

        batch_size = targets_feat.size(1)
        num_images = targets_feat.size(0)

        t = targets_feat.permute(1, 2, 0, 3, 4).contiguous() # (num_sequences, c, num_images, H, W)

        # genenrate mask
        bb = bb.clone()
        bb = (bb / feat_stride).int().view(-1, 4)
        # bb[:, :2] = bb[:, :2] + 1.0/3*bb[:, 2:]
        # bb[:, 2:] = bb[:, 2:] / 1.5
        bb[:, 2:] = bb[:, :2] + bb[:, 2:]  # (x,y,w,h) --> (x0,y0,x1,y1)
        mask = torch.ones(bb.size(0), t.size(-2), t.size(-1))
        for i in range(mask.size(0)):
            mask[i, bb[i][1]:bb[i][3], bb[i][0]:bb[i][2]] = 0
        mask = mask.view(num_images, 1, batch_size, mask.size(-2), mask.size(-1)).permute(2, 1, 0, 3, 4).contiguous().to(t.device)
        # print("mask shape: {}".format(mask.shape))

        # f = self.theta(init_filter).view(batch_size, self.inter_channels, -1)
        f = init_filter.view(batch_size, self.inter_channels, -1)
        f = f.permute(0, 2, 1)  #(b, h*w, c)

        # phi_t = self.phi(t).mul(mask).view(batch_size, self.inter_channels, -1) # (b, c, T*H*W)
        phi_t = t.mul(mask).view(batch_size, self.inter_channels, -1)  # (b, c, T*H*W)

        spa_att = torch.matmul(f, phi_t)  # (b, h*w, T*H*W)
        spa_att = torch.mean(spa_att, dim=-1).view(-1, 1, init_filter.size(-2), init_filter.size(-1)) # (b, 1, h, w)
        spa_att = 1 - torch.sigmoid(spa_att)
        # print("spa_att shape: {}".format(spa_att.shape))
        # print("init_filter shape: {}".format(init_filter.shape))

        f_att = init_filter * spa_att

        return f_att
