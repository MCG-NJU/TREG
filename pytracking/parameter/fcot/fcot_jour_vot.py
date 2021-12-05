from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone

def parameters():
    # params = TrackerParams()
    #
    # params.debug = 0
    # params.visualization = False
    #
    # params.use_gpu = True
    # params.use_classifier = True
    # params.image_sample_size = 18 * 16  # 18 * 16
    # params.search_area_scale = 4.5
    #
    # params.update_cls_using_mask = True
    #
    # # params.scale_factors = [0.975, 1.0, 1.025]
    #
    # ### Learning parameters
    # params.sample_memory_size = 50
    # params.learning_rate = 0.0075
    # params.init_samples_minimum_weight = 0.0
    # params.train_skipping = 8
    # params.init_train_frames = 15  # 15 16 这个范围最好
    #
    # ### Net optimization params
    # params.update_classifier_and_regressor = True
    # params.ues_select_sample_strategy = True
    #
    # # classifier-18
    # params.init_train_iter = 6
    # params.net_opt_iter = 25  # 10-12  23 这几个很好
    # params.net_opt_update_iter = 1
    # params.net_opt_hn_iter = 3
    #
    # ### merge the initial model and the optimized model.
    # params.lamda_72 = 0.4
    # params.lamda_cls = 1
    #
    # ### multi-scale classification
    # params.merge_rate_18 = 0.73
    #
    # ### Init augmentation parameters
    # params.use_augmentation = True
    # params.augmentation = {'fliplr': True,
    #                        'rotate': [5, -5, 10, -10, 20, -20, 30, -30, 45, -45, -60, 60],
    #                        'blur': [(2, 0.2), (0.2, 2), (3, 1), (1, 3), (2, 2)],
    #                        'relativeshift': [(0.6, 0.6), (-0.6, 0.6), (0.6, -0.6), (-0.6, -0.6)],
    #                        # 'dropout': (7, 0.2)
    #                        }
    #
    # params.augmentation_expansion_factor = 2
    # params.random_shift_factor = 1 / 3
    #
    # ### Advanced localization parameters
    # params.advanced_localization = True
    # params.target_not_found_threshold = 0
    # params.distractor_threshold = 100
    # params.hard_negative_threshold = 0.45  # 0.45
    # params.target_neighborhood_scale = 2.2
    # params.dispalcement_scale = 0.7
    #
    # params.window_output = True
    # params.perform_hn_without_windowing = True
    # params.hard_negative_learning_rate = 0.027  # 0.02
    # params.update_scale_when_uncertain = True
    #
    # params.iou_select = False
    params = TrackerParams()

    params.debug = 0
    params.visualization = False

    params.use_gpu = True
    params.use_classifier = True
    params.image_sample_size = 18 * 16
    params.search_area_scale = 4.0

    # params.scale_factors = [0.975, 1.0, 1.025]

    ### Learning parameters
    params.sample_memory_size = 250
    params.learning_rate = 0.0075
    params.init_samples_minimum_weight = 0.0
    params.train_skipping = 8
    params.init_train_frames = 8

    ### Net optimization params
    params.update_classifier_and_regressor = True
    params.ues_select_sample_strategy = True

    # classifier-18
    params.init_train_iter = 6
    params.net_opt_iter = 25
    params.net_opt_update_iter = 3
    params.net_opt_hn_iter = 3

    # classifier-72
    params.init_train_iter_72 = 6
    params.net_opt_iter_72 = 25
    params.net_opt_update_iter_72 = 3
    params.net_opt_hn_iter_72 = 3

    # # regressor
    params.reg_init_train_iter = 4
    params.reg_net_opt_iter = 10
    params.reg_net_opt_hn_iter = 1
    params.reg_net_opt_update_iter = 1

    ### merge the initial model and the optimized model.
    params.lamda_72 = 0.5
    params.lamda_cls = 1.0
    # params.reg_lamda = 0.6

    ### multi-scale classification
    params.merge_rate_72 = 0.1
    params.merge_rate_18 = 1.0

    ### Init augmentation parameters
    params.use_augmentation = True
    params.augmentation = {'fliplr': True,
                           'rotate': [5, -5, 10, -10, 20, -20, 30, -30, 45, -45, -60, 60],
                           'blur': [(2, 0.2), (0.2, 2), (3, 1), (1, 3), (2, 2)],
                           'relativeshift': [(0.6, 0.6), (-0.6, 0.6), (0.6, -0.6), (-0.6, -0.6)],
                           # 'dropout': (7, 0.2)
                           }

    params.augmentation_expansion_factor = 2
    params.random_shift_factor = 1/3

    ### Advanced localization parameters
    params.advanced_localization = True
    params.target_not_found_threshold = 0
    params.distractor_threshold = 100
    params.hard_negative_threshold = 0.45
    params.target_neighborhood_scale = 2.2
    params.dispalcement_scale = 0.7

    params.window_output = True
    params.perform_hn_without_windowing = True
    params.hard_negative_learning_rate = 0.02  # 0.02
    params.update_scale_when_uncertain = True

    params.iou_select = False

    params.reg_net = NetWithBackbone(net_path='OnlineRegressor_ep0040.pth.tar',
                                 use_gpu=params.use_gpu)
    params.reg_net.initialize()
    params.net = NetWithBackbone(net_path='ClsNet_3scales_ep0037.pth.tar',  #'FCOTNet_1s_regsz2_ep0063.pth',
                                 use_gpu=params.use_gpu)
    params.net.initialize()

    params.vot_anno_conversion_type = 'preserve_area'

    return params
