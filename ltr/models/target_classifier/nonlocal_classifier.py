import torch.nn as nn
import torch
import ltr.models.layers.filter as filter_layer
import math
from ltr.models.target_classifier.initializer import FilterPool
from ltr.models.layers.nonlocal_target import NonLocalTarget
from ltr.external.DCNv2.dcn_v2 import DCN

class NonlocalClassifier(nn.Module):
    """Target classification filter module.
    args:
        filter_size:  Size of filter (int).
        filter_initialize:  Filter initializer module.
        filter_optimizer:  Filter optimizer module.
        feature_extractor:  Feature extractor module applied to the input backbone features."""

    def __init__(self, filter_size, feature_extractor=None):
        super().__init__()

        self.filter_size = filter_size

        # Modules
        self.feature_extractor = feature_extractor
        self.filter_pool = FilterPool(filter_size=self.filter_size, feature_stride=4, pool_square=False)
        # self.filter_pool2 = FilterPool(filter_size=5, feature_stride=4, pool_square=False)  # 7 5最好
        # self.filter_pool3 = FilterPool(filter_size=9, feature_stride=4, pool_square=False)
        self.nonlocal_target = NonLocalTarget(in_channels=256, inter_channels=256, sub_sample=False, bn_layer=False)

        self.head = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            # nn.GroupNorm(32, 256),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            DCN(256, 64, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1),
            # nn.GroupNorm(32, 64),
            # nn.BatchNorm2d(64),
            nn.ReLU(),

            # DCN(64, 4, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1),
            nn.Conv2d(64, 1, 3, 1, 1, bias=False),
            # nn.ReLU(),
        )
        # self.head = nn.Sequential(
        #     nn.Conv2d(256, 64, 3, 1, 1, bias=False),
        #     # nn.GroupNorm(32, 64),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(64, 4, 3, 1, 1, bias=False),
        #     nn.ReLU(),
        # )

        # Init weights
        for m in self.feature_extractor.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.nonlocal_target.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, train_feat, test_feat, train_bb, *args, **kwargs):
        """Learns a target classification filter based on the train samples and return the resulting classification
        scores on the test samples.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_feat:  Backbone features for the train samples (4 or 5 dims).
            test_feat:  Backbone features for the test samples (4 or 5 dims).
            trian_bb:  Target boxes (x,y,w,h) for the train samples in image coordinates. Dims (images, sequences, 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            test_scores:  Classification scores on the test samples."""

        assert train_bb.dim() == 3

        bb = train_bb.clone()

        num_sequences = train_bb.shape[1]

        if train_feat.dim() == 5:
            train_feat = train_feat.view(-1, *train_feat.shape[-3:])

        if test_feat.dim() == 5:
            test_feat = test_feat.view(-1, *test_feat.shape[-3:])

        # Extract features
        train_feat = self.extract_regression_feat(train_feat, num_sequences)
        test_feat = self.extract_regression_feat(test_feat, num_sequences)

        # Train filter
        # filter, filter_iter, losses = self.get_filter(train_feat, train_bb, *args, **kwargs)

        # Classify samples using all return filters
        test_scores = self.regress(train_feat, bb, test_feat)

        return test_scores


    def extract_regression_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor is None:
            return feat
        if num_sequences is None:
            return self.feature_extractor(feat)

        output = self.feature_extractor(feat)
        return output
        # return output.view(-1, num_sequences, *output.shape[-3:])


    def regress(self, train_feat, train_bb, test_feat):
        """Run classifier (filter) on the features (feat)."""

        num_sequences = train_bb.shape[1] if train_bb.dim() == 3 else 1

        feat_pool = self.filter_pool(train_feat, train_bb)
        feat_pool = feat_pool.view(-1, num_sequences, *feat_pool.shape[-3:])

        # feat_pool2 = self.filter_pool2(train_feat, train_bb)
        # feat_pool2 = feat_pool2.view(-1, num_sequences, *feat_pool2.shape[-3:])

        test_feat = test_feat.view(-1, num_sequences, *test_feat.shape[-3:])
        feat_nonlocal, _, _ = self.nonlocal_target(feat_pool, test_feat)

        # feat_nonlocal = feat_nonlocal.view(-1, num_sequences, *feat_nonlocal.shape[-3:])
        # feat_nonlocal, _, _ = self.nonlocal_target(feat_pool2, feat_nonlocal)

        # feat_nonlocal = torch.cat([feat_nonlocal, feat_nonlocal2], dim=1)

        offset_maps = self.head(feat_nonlocal)
        offset_maps = offset_maps.view(-1, num_sequences, *offset_maps.size()[-2:])
        # print("reg maps shape: {}".format(offset_maps.shape))

        return offset_maps, feat_nonlocal