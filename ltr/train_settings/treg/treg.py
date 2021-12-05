import torch.optim as optim
import torchvision.transforms
import torch
from ltr.dataset import Lasot, Got10k, TrackingNet, MSCOCOSeq
from ltr.data import sampler, LTRLoader, processing_fcot
from ltr.models.tracking import tregnet
import ltr.models.loss as ltr_losses
from ltr.models.loss.target_regression import REGLoss
from ltr import actors
from ltr.trainers import LTRFcotTrainer
import ltr.data.transforms as dltransforms
from ltr import MultiGPU
import ltr.data.new_transform as tfm
import collections

# Address the issue: "RuntimeError: received 0 items of ancdata"
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
torch.multiprocessing.set_sharing_strategy('file_system')


def run(settings):
    settings.description = 'Default train settings for super_fcotv3, which use pretrained super-dimp, bs is 80, max_gap is 200, reg_filter_size is 7, clf_size is 4.'
    settings.use_pretrained_super_dimp = True
    settings.pretrained_super_dimp50 = "../models/super_dimp.pth.tar"
    settings.multi_gpu = True
    settings.print_interval = 1
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 6.0
    settings.output_sigma_factor = 1/4
    settings.clf_target_filter_sz = 4
    settings.reg_target_filter_sz = 3
    settings.feature_sz = 22
    settings.output_sz = settings.feature_sz * 16
    settings.center_jitter_factor = {'train': 3, 'test': 5.5}
    settings.scale_jitter_factor = {'train': 0.25, 'test': 0.5}
    settings.hinge_threshold = 0.05
    settings.logging_file = 'log.txt'

    # Train datasets
    lasot_train = Lasot(settings.env.lasot_dir, split='train')
    got10k_train = Got10k(settings.env.got10k_dir, split='vottrain')
    trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(4)))
    coco_train = MSCOCOSeq(settings.env.coco_dir)

    # Validation datasets
    got10k_val = Got10k(settings.env.got10k_dir, split='votval')


    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip(probability=0.5),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    # The tracking pairs processing module
    output_sigma = settings.output_sigma_factor / settings.search_area_factor
    label_params = {'feature_sz': settings.feature_sz, 'sigma_factor': output_sigma, 'kernel_sz': settings.clf_target_filter_sz}
    data_processing_train = processing_fcot.SuperAnchorFreeProcessing(search_area_factor=settings.search_area_factor,
                                                                      output_sz=settings.output_sz,
                                                                      output_w=88,
                                                                      output_h=88,
                                                                      center_jitter_factor=settings.center_jitter_factor,
                                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                                      mode='sequence',
                                                                      max_scale_change=1.5,
                                                                      crop_type='inside_major',
                                                                      output_spatial_scale=1 / 4.,
                                                                      label_function_params=label_params,
                                                                      transform=transform_train,
                                                                      joint_transform=transform_joint)

    data_processing_val = processing_fcot.SuperAnchorFreeProcessing(search_area_factor=settings.search_area_factor,
                                                                    output_sz=settings.output_sz,
                                                                    output_w=88,
                                                                    output_h=88,
                                                                    center_jitter_factor=settings.center_jitter_factor,
                                                                    scale_jitter_factor=settings.scale_jitter_factor,
                                                                    mode='sequence',
                                                                    crop_type='inside_major',
                                                                    max_scale_change=1.5,
                                                                    output_spatial_scale=1 / 4.,
                                                                    label_function_params=label_params,
                                                                    transform=transform_val,
                                                                    joint_transform=transform_joint)

    # Train sampler and loader
    dataset_train = sampler.FCOTSampler([lasot_train, got10k_train, trackingnet_train, coco_train], [settings.lasot_rate,1,1,1],
                                        samples_per_epoch=settings.samples_per_epoch, max_gap=200, num_test_frames=3, num_train_frames=3,
                                        processing=data_processing_train)
    # dataset_train = sampler.FCOTSampler([got10k_train],
    #                                     [1],
    #                                     samples_per_epoch=10000, max_gap=200, num_test_frames=3,
    #                                     num_train_frames=3,
    #                                     processing=data_processing_train)

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size,
                             num_workers=settings.num_workers, shuffle=True, drop_last=True, stack_dim=1)

    # Validation samplers and loaders
    dataset_val = sampler.FCOTSampler([got10k_val], [1], samples_per_epoch=10000, max_gap=200, num_test_frames=3,
                                      num_train_frames=3, processing=data_processing_val)

    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size, epoch_interval=5,
                           num_workers=settings.num_workers, shuffle=False, drop_last=True, stack_dim=1)

    # Create network
    net = tregnet.tregnet(clf_filter_size=settings.clf_target_filter_sz, reg_filter_size=settings.reg_target_filter_sz,
                          backbone_pretrained=True, optim_iter=5, norm_scale_coef=settings.norm_scale_coef,
                          clf_feat_norm=True, clf_feat_blocks=0, final_conv=True, out_feature_dim=512,
                          optim_init_step=0.9, optim_init_reg=0.1, init_gauss_sigma=output_sigma * settings.feature_sz,
                          num_dist_bins=100, bin_displacement=0.1, mask_init_factor=3.0, target_mask_act='sigmoid',
                          score_act='relu', train_reg_optimizer=settings.train_reg_optimizer,
                          train_cls_72_and_reg_init=settings.train_cls_72_and_reg_init, train_cls_18=settings.train_cls_18,
                          frozen_backbone_layers=['conv1', 'bn1', 'layer1', 'layer2'])

    # Load dimp-model as initial weights
    device = torch.device('cuda:{}'.format(settings.devices_id[0]) if torch.cuda.is_available() else 'cpu')
    if settings.use_pretrained_super_dimp:
        assert settings.pretrained_super_dimp50 is not None
        super_dimp50 = torch.load(settings.pretrained_super_dimp50, map_location=device)
        state_dict = collections.OrderedDict()
        for key, v in super_dimp50['net'].items():
            if key.split('.')[0] == 'feature_extractor':
                state_dict['.'.join(key.split('.')[1:])] = v

        net.feature_extractor.load_state_dict(state_dict)

        state_dict = collections.OrderedDict()
        for key, v in super_dimp50['net'].items():
            if key.split('.')[0] == 'classifier':
                state_dict['.'.join(key.split('.')[1:])] = v
        net.classifier_18.load_state_dict(state_dict)
        print("loading backbone and Classifier modules from super-DiMP50 done.")


    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        net = MultiGPU(net, device_ids=settings.devices_id, dim=1).to(device)

    # Loss for cls_72, cls_18 and regression
    objective = {'test_clf_72': ltr_losses.LBHinge(threshold=settings.hinge_threshold),
                 'test_clf_18': ltr_losses.LBHinge(threshold=settings.hinge_threshold),
                 'reg_72': REGLoss(dim=4)
                 }

    # Create actor and adam-optimizer
    loss_weight = {'test_clf_72': 100, 'test_init_clf_72': 100, 'test_iter_clf_72': 400,
                    'test_clf_18': 100, 'test_init_clf_18': 100, 'test_iter_clf_18': 400,
                    'reg_72': 1}
    actor = actors.FcotActor(net=net, objective=objective, loss_weight=loss_weight, device=device)
    optimizer = optim.Adam([{'params': actor.net.classifier_72.filter_initializer.parameters(), 'lr': 5e-5},
                            {'params': actor.net.classifier_72.filter_optimizer.parameters(), 'lr': 5e-4},
                            {'params': actor.net.classifier_72.feature_extractor.parameters(), 'lr': 5e-5},
                            {'params': actor.net.classifier_18.filter_initializer.parameters(), 'lr': 5e-5},
                            {'params': actor.net.classifier_18.filter_optimizer.parameters(), 'lr': 5e-4},
                            {'params': actor.net.classifier_18.feature_extractor.parameters(), 'lr': 5e-5},
                            {'params': actor.net.regressor_72.parameters()},
                            {'params': actor.net.pyramid_first_conv.parameters()},
                            {'params': actor.net.pyramid_36.parameters()},
                            {'params': actor.net.pyramid_72.parameters()},
                            {'params': actor.net.feature_extractor.parameters(), 'lr': 2e-5}
                            ],
                            lr=2e-4)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[28, 35, 45], gamma=0.2)

    trainer = LTRFcotTrainer(actor, [loader_train, loader_val], optimizer, settings, device, lr_scheduler,
                             logging_file=settings.logging_file)

    trainer.train(settings.total_epochs, load_latest=True, fail_safe=True)