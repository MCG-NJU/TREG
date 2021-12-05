import math
import torch.nn as nn
from collections import OrderedDict
import ltr.models.target_classifier.linear_filter as target_clf
import ltr.models.target_classifier.features as clf_features
import ltr.models.target_classifier.initializer as clf_initializer
import ltr.models.target_classifier.optimizer as clf_optimizer
import ltr.models.backbone as backbones
from ltr.models.target_classifier.up_features import FPNUpBlock
from ltr import model_constructor
from ltr.models.target_regression.nonlocal_regressor import NonlocalRegressor
import os, cv2

class TREGNet(nn.Module):
    """The FCOTNet network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps.
        classification_layer:  Name of the backbone feature layer to use for classification.
        pyramid_*: The blocks in Decoder.
        classifier_18:  Target classification module with feature size of 18.
        classifier_72:  Target classification module with feature size of 72.
        regressor_72:  Anchor-free regression module with feature size of 72.
        train_cls_72_and_reg_init: Whether clssifier_72 and regression initial model should be trained or not.
        train_reg_optimizer:  Whether regression optimizer should be trained or not.
        train_cls_18:  Whether classifier_18 should be trained or not. """

    def __init__(self, feature_extractor, classification_layer, pyramid_first_conv, pyramid_36, pyramid_72, classifier_18,
                 classifier_72, regressor_72):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.classifier_18 = classifier_18
        self.classifier_72 = classifier_72
        self.regressor_72 = regressor_72
        self.pyramid_first_conv = pyramid_first_conv
        self.pyramid_36 = pyramid_36
        self.pyramid_72 = pyramid_72


        self.classification_layer = [classification_layer] if isinstance(classification_layer, str) else classification_layer
        self.output_layers = ['layer1', 'layer2', 'layer3']


    def forward(self, train_imgs, test_imgs, train_bb, *args, **kwargs):

        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'

        # Extract backbone features
        train_feat = self.extract_backbone_features(train_imgs.view(-1, *train_imgs.shape[-3:]),
                                                    layers=['layer1','layer2','layer3'])
        test_feat = self.extract_backbone_features(test_imgs.view(-1, *test_imgs.shape[-3:]),
                                                   layers=['layer1','layer2','layer3'])

        # Classification features
        train_feat_clf = self.get_backbone_clf_feat(train_feat)
        test_feat_clf = self.get_backbone_clf_feat(test_feat)

        # generate score_map_18
        target_scores_18 = self.classifier_18(train_feat_clf, test_feat_clf, train_bb, *args, **kwargs)

        # Extract features using decoder
        train_feat_18 = self.pyramid_first_conv(x=None, x_backbone=train_feat_clf)
        test_feat_18 = self.pyramid_first_conv(x=None, x_backbone=test_feat_clf)

        train_feat_36 = self.pyramid_36(train_feat_18, train_feat['layer2'])
        test_feat_36 = self.pyramid_36(test_feat_18, test_feat['layer2'])

        train_feat_72 = self.pyramid_72(train_feat_36, train_feat['layer1'])
        test_feat_72 = self.pyramid_72(test_feat_36, test_feat['layer1'])

        # generate score_map_72 and offset_maps
        target_scores_72 = self.classifier_72(train_feat_72, test_feat_72, train_bb, *args, **kwargs)
        offset_maps = self.regressor_72(train_feat_72, test_feat_72, train_bb, *args, **kwargs)


        return target_scores_18, target_scores_72, offset_maps

    def get_backbone_clf_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.classification_layer})
        if len(self.classification_layer) == 1:
            return feat[self.classification_layer[0]]
        return feat

    def extract_classification_feat_18(self, backbone_feat):
        return self.classifier_18.extract_classification_feat(self.get_backbone_clf_feat(backbone_feat))

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers=None):
        if layers is None:
            layers = self.bb_regressor_layer + ['classification']
        if 'classification' not in layers:
            return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.classification_layer if l != 'classification'])))
        all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.extract_classification_feat(all_feat)
        return OrderedDict({l: all_feat[l] for l in layers})


@model_constructor
def tregnet(clf_filter_size=4, reg_filter_size=3, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
            classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=0,
            clf_feat_norm=True, init_filter_norm=False, final_conv=True, out_feature_dim=512,
            init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0, mask_init_factor=4.0,
            score_act='relu', act_param=None, target_mask_act='sigmoid', detach_length=float('Inf'),
            frozen_backbone_layers=()):
    # Backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    pyramid_first_conv = FPNUpBlock(res_channels=1024, planes=256, smooth_output=False, first_conv=True)

    up_36 = FPNUpBlock(res_channels=512, planes=256, smooth_output=False, first_conv=False)

    up_72 = FPNUpBlock(res_channels=256, planes=256, smooth_output=True, first_conv=False)

    # classifier_72
    norm_scale_72 = math.sqrt(2 / (256 * 4 * 4))
    clf_head_72 = clf_features.clf_head_72(feature_dim=256, l2norm=clf_feat_norm, norm_scale=norm_scale_72,
                                           out_dim=256)
    initializer_72 = clf_initializer.FilterInitializerLinear(filter_size=4, filter_norm=init_filter_norm,
                                                             feature_dim=256, feature_stride=4)
    optimizer_72 = clf_optimizer.DiMPSteepestDescentGN(num_iter=optim_iter, feat_stride=4,
                                                       init_step_length=optim_init_step,
                                                       init_filter_reg=optim_init_reg,
                                                       init_gauss_sigma=init_gauss_sigma,
                                                       num_dist_bins=num_dist_bins,
                                                       bin_displacement=bin_displacement,
                                                       mask_init_factor=mask_init_factor,
                                                       score_act=score_act, act_param=act_param,
                                                       mask_act=target_mask_act,
                                                       detach_length=detach_length)
    classifier_72 = target_clf.LinearFilter(filter_size=4, filter_initializer=initializer_72,
                                            filter_optimizer=optimizer_72, feature_extractor=clf_head_72)

    # classifier_18 (We use the same architecture of classifier_18 with DiMP.)
    norm_scale_18 = math.sqrt(1.0 / (out_feature_dim * clf_filter_size * clf_filter_size))
    clf_head_18 = clf_features.clf_head_18(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                           final_conv=final_conv, norm_scale=norm_scale_18,
                                           out_dim=out_feature_dim)
    initializer_18 = clf_initializer.FilterInitializerLinear(filter_size=clf_filter_size, filter_norm=init_filter_norm,
                                                             feature_dim=out_feature_dim)
    optimizer_18 = clf_optimizer.DiMPSteepestDescentGN(num_iter=optim_iter, feat_stride=feat_stride,
                                                       init_step_length=optim_init_step,
                                                       init_filter_reg=optim_init_reg,
                                                       init_gauss_sigma=init_gauss_sigma,
                                                       num_dist_bins=num_dist_bins,
                                                       bin_displacement=bin_displacement,
                                                       mask_init_factor=mask_init_factor,
                                                       score_act=score_act, act_param=act_param,
                                                       mask_act=target_mask_act,
                                                       detach_length=detach_length)
    classifier_18 = target_clf.LinearFilter(filter_size=clf_filter_size, filter_initializer=initializer_18,
                                            filter_optimizer=optimizer_18, feature_extractor=clf_head_18)

    # regressor_72
    reg_head_72 = clf_features.head_72(feature_dim=256, l2norm=False, norm_scale=norm_scale_72,
                                       out_dim=256, inner_dim=256)
    regressor_72 = NonlocalRegressor(filter_size=5, feature_extractor=reg_head_72)

    # TREG network
    net = TREGNet(feature_extractor=backbone_net, classification_layer=classification_layer,
                  pyramid_first_conv=pyramid_first_conv, pyramid_36=up_36, pyramid_72=up_72,
                  classifier_18=classifier_18,
                  classifier_72=classifier_72, regressor_72=regressor_72)
    return net