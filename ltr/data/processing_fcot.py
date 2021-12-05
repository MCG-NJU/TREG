import torch
import torchvision.transforms as transforms
from pytracking import TensorDict
import ltr.data.processing_utils as prutils
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class BaseProcessing:
    def __init__(self, transform=transforms.ToTensor(), train_transform=None, test_transform=None, joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if train_transform or
                                test_transform is None.
            train_transform - The set of transformations to be applied on the train images. If None, the 'transform'
                                argument is used instead.
            test_transform  - The set of transformations to be applied on the test images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the train and test images.  For
                                example, it can be used to convert both test and train images to grayscale.
        """
        self.transform = {'train': transform if train_transform is None else train_transform,
                          'test':  transform if test_transform is None else test_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class AnchorFreeProcessing(BaseProcessing):

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, crop_type='replicate',
                 mode='pair', output_spatial_scale=1/4., output_w=72, output_h=72,
                 proposal_params=None, label_function_params=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            crop_type - If 'replicate', the boundary pixels are replicated in case the search region crop goes out of image.
                        If 'nopad', the search region crop is shifted/shrunk to fit completely inside the image.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
            proposal_params - Arguments for the proposal generation process. See _generate_proposals for details.
            label_function_params - Arguments for the label generation process. See _generate_label_function for details.
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.crop_type = crop_type
        self.mode = mode

        self.proposal_params = proposal_params
        self.label_function_params = label_function_params

        self.output_w = output_w
        self.output_h = output_h
        self.output_spatial_scale = output_spatial_scale

        self.count = 0

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * self.center_jitter_factor[mode]).item()
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_proposals(self, box):
        """ Generates proposals by adding noise to the input box
        args:
            box - input box

        returns:
            torch.Tensor - Array of shape (num_proposals, 4) containing proposals
            torch.Tensor - Array of shape (num_proposals,) containing IoU overlap of each proposal with the input box. The
                        IoU is mapped to [-1, 1]
        """
        # Generate proposals
        num_proposals = self.proposal_params['boxes_per_frame']
        proposals = torch.zeros((num_proposals, 4))
        gt_iou = torch.zeros(num_proposals)

        for i in range(num_proposals):
            proposals[i, :], gt_iou[i] = prutils.perturb_box(box, min_iou=self.proposal_params['min_iou'],
                                                             sigma_factor=self.proposal_params['sigma_factor'])

        # Map to [-1, 1]
        gt_iou = gt_iou * 2 - 1
        return proposals, gt_iou

    def _generate_label_function(self, target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """

        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), self.label_function_params['sigma_factor'],
                                                      self.label_function_params['kernel_sz'],
                                                      self.label_function_params['feature_sz'], self.output_sz,
                                                      end_pad_if_even=self.label_function_params.get('end_pad_if_even', True))

        return gauss_label

    def _generate_label_72_function(self, target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """

        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), 1 / 40.,
                                                      self.label_function_params['kernel_sz'],
                                                      self.output_sz * self.output_spatial_scale, self.output_sz,
                                                      end_pad_if_even=self.label_function_params.get('end_pad_if_even',
                                                                                                     True))

        return gauss_label


    def save_groundtruth_image(self, img, bbox, save_file = None):
        """Display a 2D tensor.
        args:
            fig_num: Figure number.
            title: Title of figure.
        """

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        draw_rec = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
        cv2.imwrite(save_file, draw_rec)

    def save_test_image(self, img,save_file = None):
        """Display a 2D tensor.
        args:
            fig_num: Figure number.
            title: Title of figure.
        """

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_file, img)

    def save_tensor_as_image(self, a, save_file=None, range=(None, None)):
        """Display a 2D tensor.
        args:
            fig_num: Figure number.
            title: Title of figure.
        """
        if a.ndim == 3:
            a = np.transpose(a, (1, 2, 0))
        plt.imsave(save_file, a)

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'train_images'  -
                'test_images'   -
                'train_anno'    -
                'test_anno'     -
        """

        if self.transform['joint'] is not None:
            num_train_images = len(data['train_images'])
            all_images = data['train_images'] + data['test_images']
            all_images_trans = self.transform['joint'](*all_images)

            data['train_images'] = all_images_trans[:num_train_images]
            data['test_images'] = all_images_trans[num_train_images:]

        # super dimp settings
        # if self.transform['joint'] is not None:
        #     data['train_images'], data['train_anno'] = self.transform['joint'](image=data['train_images'],
        #                                                                        bbox=data['train_anno'])
        #     data['test_images'], data['test_anno'] = self.transform['joint'](image=data['test_images'],
        #                                                                      bbox=data['test_anno'], new_roll=False)

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            if self.crop_type == 'replicate':
                crops, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                            self.search_area_factor, self.output_sz)
            elif self.crop_type == 'nopad':
                crops, boxes = prutils.jittered_center_crop_nopad(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                                  self.search_area_factor, self.output_sz)
            else:
                raise ValueError('Unknown crop type {}'.format(self.crop_type))

            data[s + '_images'] = [self.transform[s](x) for x in crops]


            boxes = torch.stack(boxes)
            boxes_init = boxes
            boxes_init[:, 2:4] = boxes[:, 0:2] + boxes[:, 2:4]
            boxes = boxes_init.clamp(0.0, self.output_sz - 1)

            boxes[:, 2:4] = boxes[:, 2:4] - boxes[:, 0:2]
            data[s + '_anno'] = boxes

        if self.proposal_params:
            frame2_proposals, gt_iou = zip(*[self._generate_proposals(a) for a in data['test_anno']])

            data['test_proposals'] = list(frame2_proposals)
            data['proposal_iou'] = list(gt_iou)

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(prutils.stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

    #     boxes_out_xywh = (data['test_anno'].float() * self.output_spatial_scale).float()
    #     center_float = torch.stack((boxes_out_xywh[:, 0] + boxes_out_xywh[:, 2] / 2.,
    #                                 boxes_out_xywh[:, 1] + boxes_out_xywh[:, 3] / 2.), dim=1)
    #     center_int = center_float.int().float()
    #     # boxes is in format xywh, convert it to x0y0x1y1 format
    #     boxes_out_xyxy = boxes_out_xywh.clone()
    #     boxes_out_xyxy[:, 2:] = boxes_out_xyxy[:, :2] + boxes_out_xyxy[:, 2:]
    #
    #     reg_labels = self._generate_regression_label(boxes_out_xyxy, size=self.output_w,
    #                                                  spatial_scale=self.output_spatial_scale)
    #     reg_mask = self._generate_regression_mask(center_int, reg_labels=reg_labels, radius=2, mask_size=self.output_w)
    #     # centerness = self._generate_centerness_label(reg_labels)
    #
    #     data['reg_labels'] = reg_labels
    #     data['reg_mask'] = reg_mask
    #
    #     # ind_out = center_int[:, 1] * self.output_w + center_int[:, 0]  # [num_images, 1]
    #     # data['ind_out'] = ind_out.long()
    #
    #     ### Generate classification labels
    #     if self.label_function_params is not None:
    #         # data['train_label'] = self._generate_label_function(data['train_anno'])
    #         data['test_label'] = self._generate_label_function(data['test_anno'])
    #         data['test_label_72'] = self._generate_label_72_function(data['test_anno'])
    #         # data['reg_gauss_center_label'] = self._generate_gauss_reg_label_function(data['test_anno'])
    #
    #     return data
    #
    # def _generate_regression_mask(self, target_center, reg_labels=None, radius=2, mask_size=72):
    #     """
    #     NHW format
    #     :param target_center:
    #     :param radius:
    #     :param mask_size:
    #     :return:
    #     """
    #     k0 = torch.arange(mask_size, dtype=torch.float32, device=target_center.device).view(1, 1, -1)
    #     k1 = torch.arange(mask_size, dtype=torch.float32, device=target_center.device).view(1, -1, 1)
    #
    #     d0 = k0 - target_center[:, 0].view(-1, 1, 1)
    #     d1 = k1 - target_center[:, 1].view(-1, 1, 1)
    #     dist = d0.abs() + d1.abs()
    #
    #     rPos = radius + 0.1
    #     mask = torch.where(dist <= rPos, torch.ones_like(dist), torch.zeros_like(dist))  # (n, size, size)
    #
    #     # remove the points outside the target box.
    #     if reg_labels is not None:
    #         mask = mask.unsqueeze(1).repeat(1, reg_labels.size(1), 1, 1)
    #         mask = torch.where(reg_labels <= 0, torch.zeros_like(mask), mask)
    #         mask = mask.min(dim=1)[0]  # (n, size, size)
    #
    #     return mask
    #
    # def _generate_centerness_label(self, reg_labels):
    #     left_right = reg_labels[:, :2, ...]
    #     top_bottom = reg_labels[:, 2:, ...]
    #     centerness = (left_right.min(dim=1)[0] / left_right.max(dim=1)[0]) * \
    #                  (top_bottom.min(dim=1)[0] / top_bottom.max(dim=1)[0])  # (n, size, size)
    #     centerness = centerness.clamp(min=0.0)
    #     return torch.sqrt(centerness)
    #
    # def _generate_regression_label(self, boxes_xyxy, size=72, spatial_scale=1.0):
    #     x = torch.arange(size, dtype=torch.float32, device=boxes_xyxy.device).view(1, 1, -1)
    #     y = torch.arange(size, dtype=torch.float32, device=boxes_xyxy.device).view(1, -1, 1)
    #
    #     l = (x - boxes_xyxy[:, 0].view(-1, 1, 1)).repeat(1, size,
    #                                                      1)  # / spatial_scale # (n, 1, size) --> (n, size, size)
    #     r = (boxes_xyxy[:, 2].view(-1, 1, 1) - x).repeat(1, size,
    #                                                      1)  # / spatial_scale # (n, 1, size) --> (n, size, size)
    #     t = (y - boxes_xyxy[:, 1].view(-1, 1, 1)).repeat(1, 1,
    #                                                      size)  # / spatial_scale # (n, size, 1) --> (n, size, size)
    #     b = (boxes_xyxy[:, 3].view(-1, 1, 1) - y).repeat(1, 1,
    #                                                      size)  # / spatial_scale # (n, size, 1) --> (n, size, size)
    #
    #     reg_labels = torch.stack([l, r, t, b], dim=1)  # (n, 4, size, size)
    #     return reg_labels

        test_anno = data['test_anno'].clone()
        test_anno[:, 2:4] = test_anno[:, 0:2] + test_anno[:, 2:4]
        center_288 = (test_anno[:, 0:2] + test_anno[:, 2:4]) * 0.5
        # w_288, h_288 = test_anno[:, 2] - test_anno[:, 0], test_anno[:, 3] - test_anno[:, 1]
        wl_288, wr_288 = center_288[:, 0] - test_anno[:, 0], test_anno[:, 2] - center_288[:, 0]
        ht_288, hb_288 = center_288[:, 1] - test_anno[:, 1], test_anno[:, 3] - center_288[:, 1]
        w2h2_288 = torch.stack((wl_288, wr_288, ht_288, hb_288), dim=1)  # [num_images, 4]

        boxes_72 = (data['test_anno'] * self.output_spatial_scale).float()
        # boxes is in format xywh, convert it to x0y0x1y1 format
        boxes_72[:, 2:4] = boxes_72[:, 0:2] + boxes_72[:, 2:4]

        center_float = torch.stack(((boxes_72[:, 0] + boxes_72[:, 2]) / 2., (boxes_72[:, 1] + boxes_72[:, 3]) / 2.), dim=1)
        center_int = center_float.int().float()
        ind_72 = center_int[:, 1] * self.output_w + center_int[:, 0]  # [num_images, 1]

        data['ind_72'] = ind_72.long()
        data['w2h2_288'] = w2h2_288
        data['w2h2_72'] = w2h2_288 * self.output_spatial_scale

        ### Generate label functions
        if self.label_function_params is not None:
            # data['train_label'] = self._generate_label_function(data['train_anno'])
            data['test_label'] = self._generate_label_function(data['test_anno'])

            # data['train_label_72'] = self._generate_label_72_function(data['train_anno'])
            data['test_label_72'] = self._generate_label_72_function(data['test_anno'])

        return data



class FCOTProcessing(BaseProcessing):

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, crop_type='replicate',
                 mode='pair', output_spatial_scale=1/16., output_w=18, output_h=18,
                 proposal_params=None, label_function_params=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            crop_type - If 'replicate', the boundary pixels are replicated in case the search region crop goes out of image.
                        If 'nopad', the search region crop is shifted/shrunk to fit completely inside the image.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
            proposal_params - Arguments for the proposal generation process. See _generate_proposals for details.
            label_function_params - Arguments for the label generation process. See _generate_label_function for details.
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.crop_type = crop_type
        self.mode = mode

        self.proposal_params = proposal_params
        self.label_function_params = label_function_params

        self.output_w = output_w
        self.output_h = output_h
        self.output_spatial_scale = output_spatial_scale

        self.count = 0

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * self.center_jitter_factor[mode]).item()
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_proposals(self, box):
        """ Generates proposals by adding noise to the input box
        args:
            box - input box

        returns:
            torch.Tensor - Array of shape (num_proposals, 4) containing proposals
            torch.Tensor - Array of shape (num_proposals,) containing IoU overlap of each proposal with the input box. The
                        IoU is mapped to [-1, 1]
        """
        # Generate proposals
        num_proposals = self.proposal_params['boxes_per_frame']
        proposals = torch.zeros((num_proposals, 4))
        gt_iou = torch.zeros(num_proposals)

        for i in range(num_proposals):
            proposals[i, :], gt_iou[i] = prutils.perturb_box(box, min_iou=self.proposal_params['min_iou'],
                                                             sigma_factor=self.proposal_params['sigma_factor'])

        # Map to [-1, 1]
        gt_iou = gt_iou * 2 - 1
        return proposals, gt_iou

    def _generate_label_function(self, target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """

        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), self.label_function_params['sigma_factor'],
                                                      self.label_function_params['kernel_sz'],
                                                      self.label_function_params['feature_sz'], self.output_sz,
                                                      end_pad_if_even=self.label_function_params.get('end_pad_if_even', True))

        return gauss_label

    def _generate_gauss_reg_label_function(self, target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """

        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), 1 / 40.,
                                                      3,
                                                      self.output_sz * self.output_spatial_scale, self.output_sz,
                                                      end_pad_if_even=self.label_function_params.get('end_pad_if_even',
                                                                                                     True))

        return gauss_label

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'train_images'  -
                'test_images'   -
                'train_anno'    -
                'test_anno'     -
        """

        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'] = self.transform['joint'](image=data['train_images'],
                                                                               bbox=data['train_anno'])
            data['test_images'], data['test_anno'] = self.transform['joint'](image=data['test_images'],
                                                                             bbox=data['test_anno'], new_roll=False)

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            if self.crop_type == 'replicate':
                crops, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                            self.search_area_factor, self.output_sz)
            elif self.crop_type == 'nopad':
                crops, boxes = prutils.jittered_center_crop_nopad(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                                  self.search_area_factor, self.output_sz)
            else:
                raise ValueError('Unknown crop type {}'.format(self.crop_type))

            data[s + '_images'] = [self.transform[s](x) for x in crops]

            data[s + '_images'], boxes = self.transform[s](image=crops, bbox=boxes, joint=False)

            boxes = torch.stack(boxes)
            boxes_init = boxes
            boxes_init[:, 2:4] = boxes[:, 0:2] + boxes[:, 2:4]
            boxes = boxes_init.clamp(0.0, self.output_sz - 1)

            boxes[:, 2:4] = boxes[:, 2:4] - boxes[:, 0:2]
            data[s + '_anno'] = boxes

        if self.proposal_params:
            frame2_proposals, gt_iou = zip(*[self._generate_proposals(a) for a in data['test_anno']])

            data['test_proposals'] = list(frame2_proposals)
            data['proposal_iou'] = list(gt_iou)

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(prutils.stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        ### Generate regression labels
        # w2h2 label
        # test_anno = data['test_anno'].clone()
        # test_anno[:, 2:4] = test_anno[:, 0:2] + test_anno[:, 2:4]
        # center = (test_anno[:, 0:2] + test_anno[:, 2:4]) * 0.5
        # wl, wr = center[:, 0] - test_anno[:, 0], test_anno[:, 2] - center[:, 0]
        # ht, hb = center[:, 1] - test_anno[:, 1], test_anno[:, 3] - center[:, 1]
        # w2h2 = torch.stack((wl, wr, ht, hb), dim=1)  # [num_images, 4]
        # data['w2h2'] = w2h2
        #
        # # indice label of output feature
        # boxes_out = (data['test_anno'] * self.output_spatial_scale).float()
        # # boxes is in format xywh, convert it to x0y0x1y1 format
        # boxes_out[:, 2:4] = boxes_out[:, 0:2] + boxes_out[:, 2:4]
        # center_float = torch.stack(
        #     ((boxes_out[:, 0] + boxes_out[:, 2]) / 2., (boxes_out[:, 1] + boxes_out[:, 3]) / 2.), dim=1)
        # center_int = center_float.int().float()
        # ind_out = center_int[:, 1] * self.output_w + center_int[:, 0]  # [num_images, 1]
        # data['ind_out'] = ind_out.long()

        ### Generate regression labels
        # w2h2 label
        # test_anno = data['test_anno'].clone()
        # test_anno[:, 2:4] = test_anno[:, 0:2] + test_anno[:, 2:4]
        # center = (test_anno[:, 0:2] + test_anno[:, 2:4]) * 0.5
        # wl, wr = center[:, 0] - test_anno[:, 0], test_anno[:, 2] - center[:, 0]
        # ht, hb = center[:, 1] - test_anno[:, 1], test_anno[:, 3] - center[:, 1]
        # w2h2 = torch.stack((wl, wr, ht, hb), dim=1)  # [num_images, 4]
        # data['w2h2'] = w2h2

        # indice label of output feature
        # boxes_out_xywh = (data['test_anno'] * self.output_spatial_scale).float()
        # center_float = torch.stack((boxes_out_xywh[:, 0] + boxes_out_xywh[:, 2] / 2.,
        #                             boxes_out_xywh[:, 1] + boxes_out_xywh[:, 3] / 2.), dim=1)
        # center_int = center_float.int().float()
        # # boxes is in format xywh, convert it to x0y0x1y1 format
        # boxes_out_xyxy = boxes_out_xywh.clone()
        # boxes_out_xyxy[:, 2:] = boxes_out_xyxy[:, :2] + boxes_out_xyxy[:, 2:]
        #
        # reg_labels = self._generate_regression_label(boxes_out_xyxy, size=72, spatial_scale=self.output_spatial_scale)
        # reg_mask = self._generate_regression_mask(center_int, reg_labels=reg_labels, radius=2, mask_size=72)
        # # centerness = self._generate_centerness_label(reg_labels)
        #
        # data['reg_labels'] = reg_labels
        # data['reg_mask'] = reg_mask
        # data['centerness'] = centerness

        boxes_out_xywh = (data['test_anno'].float() * self.output_spatial_scale).float()
        center_float = torch.stack((boxes_out_xywh[:, 0] + boxes_out_xywh[:, 2] / 2.,
                                    boxes_out_xywh[:, 1] + boxes_out_xywh[:, 3] / 2.), dim=1)
        center_int = center_float.int().float()
        # boxes is in format xywh, convert it to x0y0x1y1 format
        boxes_out_xyxy = boxes_out_xywh.clone()
        boxes_out_xyxy[:, 2:] = boxes_out_xyxy[:, :2] + boxes_out_xyxy[:, 2:]

        reg_labels = self._generate_regression_label(boxes_out_xyxy, size=self.output_w, spatial_scale=self.output_spatial_scale)
        reg_mask = self._generate_regression_mask(center_int, reg_labels=reg_labels, radius=2, mask_size=self.output_w)
        # centerness = self._generate_centerness_label(reg_labels)

        data['reg_labels'] = reg_labels
        data['reg_mask'] = reg_mask

        # ind_out = center_int[:, 1] * self.output_w + center_int[:, 0]  # [num_images, 1]
        # data['ind_out'] = ind_out.long()

        ### Generate classification labels
        if self.label_function_params is not None:
            # data['train_label'] = self._generate_label_function(data['train_anno'])
            data['test_label'] = self._generate_label_function(data['test_anno'])
            data['test_label_72'] = self._generate_gauss_reg_label_function(data['test_anno'])
            # data['reg_gauss_center_label'] = self._generate_gauss_reg_label_function(data['test_anno'])

        return data

    def _generate_regression_mask(self, target_center, reg_labels=None, radius=2, mask_size=72):
        """
        NHW format
        :param target_center:
        :param radius:
        :param mask_size:
        :return:
        """
        k0 = torch.arange(mask_size, dtype=torch.float32, device=target_center.device).view(1, 1, -1)
        k1 = torch.arange(mask_size, dtype=torch.float32, device=target_center.device).view(1, -1, 1)

        d0 = k0 - target_center[:, 0].view(-1, 1, 1)
        d1 = k1 - target_center[:, 1].view(-1, 1, 1)
        dist = d0.abs() + d1.abs()

        rPos = radius + 0.1
        mask = torch.where(dist <= rPos, torch.ones_like(dist), torch.zeros_like(dist)) # (n, size, size)

        # remove the points outside the target box.
        if reg_labels is not None:
            mask = mask.unsqueeze(1).repeat(1, reg_labels.size(1), 1, 1)
            mask = torch.where(reg_labels<=0, torch.zeros_like(mask), mask)
            mask = mask.min(dim=1)[0]  # (n, size, size)

        return mask

    def _generate_centerness_label(self, reg_labels):
        left_right = reg_labels[:, :2, ...]
        top_bottom = reg_labels[:, 2:, ...]
        centerness = (left_right.min(dim=1)[0] / left_right.max(dim=1)[0]) * \
                     (top_bottom.min(dim=1)[0] / top_bottom.max(dim=1)[0])  # (n, size, size)
        centerness = centerness.clamp(min=0.0)
        return torch.sqrt(centerness)

    def _generate_regression_label(self, boxes_xyxy, size=72, spatial_scale=1.0):
        x = torch.arange(size, dtype=torch.float32, device=boxes_xyxy.device).view(1, 1, -1)
        y = torch.arange(size, dtype=torch.float32, device=boxes_xyxy.device).view(1, -1, 1)

        l = (x - boxes_xyxy[:, 0].view(-1, 1, 1)).repeat(1, size, 1) #/ spatial_scale # (n, 1, size) --> (n, size, size)
        r = (boxes_xyxy[:, 2].view(-1, 1, 1) - x).repeat(1, size, 1) #/ spatial_scale # (n, 1, size) --> (n, size, size)
        t = (y - boxes_xyxy[:, 1].view(-1, 1, 1)).repeat(1, 1, size) #/ spatial_scale # (n, size, 1) --> (n, size, size)
        b = (boxes_xyxy[:, 3].view(-1, 1, 1) - y).repeat(1, 1, size) #/ spatial_scale # (n, size, 1) --> (n, size, size)

        reg_labels = torch.stack([l,r,t,b], dim=1)  # (n, 4, size, size)
        return reg_labels


class SuperAnchorFreeProcessing(BaseProcessing):

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, crop_type='replicate',
                 mode='pair', output_spatial_scale=1/4., output_w=72, output_h=72, max_scale_change=None,
                 proposal_params=None, label_function_params=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            crop_type - If 'replicate', the boundary pixels are replicated in case the search region crop goes out of image.
                        If 'nopad', the search region crop is shifted/shrunk to fit completely inside the image.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
            proposal_params - Arguments for the proposal generation process. See _generate_proposals for details.
            label_function_params - Arguments for the label generation process. See _generate_label_function for details.
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.crop_type = crop_type
        self.mode = mode

        self.label_function_params = label_function_params

        self.output_w = output_w
        self.output_h = output_h
        self.output_spatial_scale = output_spatial_scale

        self.max_scale_change = max_scale_change

        self.count = 0

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """
        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * self.center_jitter_factor[mode]).item()
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)


    def _generate_label_function(self, target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """
        #####  ！！！！kernel_sz后面需要改一下，现在写死了为4
        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), self.label_function_params['sigma_factor'],
                                                      4,
                                                      self.label_function_params['feature_sz'], self.output_sz,
                                                      end_pad_if_even=self.label_function_params.get('end_pad_if_even', True))

        return gauss_label

    def _generate_label_72_function(self, target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """

        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), 1 / 40.,
                                                      self.label_function_params['kernel_sz'],
                                                      self.output_sz * self.output_spatial_scale, self.output_sz,
                                                      end_pad_if_even=self.label_function_params.get('end_pad_if_even',
                                                                                                     True))

        return gauss_label


    def _generate_label_36_function(self, target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """

        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), 1 / 30.,
                                                      self.label_function_params['kernel_sz'],
                                                      self.output_sz * self.output_spatial_scale / 2, self.output_sz,
                                                      end_pad_if_even=self.label_function_params.get('end_pad_if_even',
                                                                                                     True))

        return gauss_label


    def save_groundtruth_image(self, img, bbox, save_file = None):
        """Display a 2D tensor.
        args:
            fig_num: Figure number.
            title: Title of figure.
        """

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        draw_rec = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
        cv2.imwrite(save_file, draw_rec)

    def save_test_image(self, img,save_file = None):
        """Display a 2D tensor.
        args:
            fig_num: Figure number.
            title: Title of figure.
        """

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_file, img)

    def save_tensor_as_image(self, a, save_file=None, range=(None, None)):
        """Display a 2D tensor.
        args:
            fig_num: Figure number.
            title: Title of figure.
        """
        if a.ndim == 3:
            a = np.transpose(a, (1, 2, 0))
        plt.imsave(save_file, a)

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'train_images'  -
                'test_images'   -
                'train_anno'    -
                'test_anno'     -
        """

        # if self.transform['joint'] is not None:
        #     num_train_images = len(data['train_images'])
        #     all_images = data['train_images'] + data['test_images']
        #     all_images_trans = self.transform['joint'](*all_images)
        #
        #     data['train_images'] = all_images_trans[:num_train_images]
        #     data['test_images'] = all_images_trans[num_train_images:]

        # super dimp settings
        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'] = self.transform['joint'](image=data['train_images'],
                                                                               bbox=data['train_anno'])
            data['test_images'], data['test_anno'] = self.transform['joint'](image=data['test_images'],
                                                                             bbox=data['test_anno'], new_roll=False)

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # if self.crop_type == 'replicate':
            #     crops, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
            #                                                 self.search_area_factor, self.output_sz)
            # elif self.crop_type == 'nopad':
            #     crops, boxes = prutils.jittered_center_crop_nopad(data[s + '_images'], jittered_anno, data[s + '_anno'],
            #                                                       self.search_area_factor, self.output_sz)
            # else:
            #     raise ValueError('Unknown crop type {}'.format(self.crop_type))
            #
            # data[s + '_images'] = [self.transform[s](x) for x in crops]

            crops, boxes = prutils.target_image_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                     self.search_area_factor, self.output_sz, mode=self.crop_type,
                                                     max_scale_change=self.max_scale_change)

            data[s + '_images'], boxes = self.transform[s](image=crops, bbox=boxes, joint=False)

            boxes = torch.stack(boxes)
            boxes_init = boxes
            boxes_init[:, 2:4] = boxes[:, 0:2] + boxes[:, 2:4]
            boxes = boxes_init.clamp(0.0, self.output_sz - 1)

            boxes[:, 2:4] = boxes[:, 2:4] - boxes[:, 0:2]
            data[s + '_anno'] = boxes

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(prutils.stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        test_anno = data['test_anno'].clone()
        test_anno[:, 2:4] = test_anno[:, 0:2] + test_anno[:, 2:4]
        center_288 = (test_anno[:, 0:2] + test_anno[:, 2:4]) * 0.5
        w_288, h_288 = test_anno[:, 2] - test_anno[:, 0], test_anno[:, 3] - test_anno[:, 1]
        wl_288, wr_288 = center_288[:, 0] - test_anno[:, 0], test_anno[:, 2] - center_288[:, 0]
        ht_288, hb_288 = center_288[:, 1] - test_anno[:, 1], test_anno[:, 3] - center_288[:, 1]
        w2h2_288 = torch.stack((wl_288, wr_288, ht_288, hb_288), dim=1)  # [num_images, 4]

        boxes_72 = (data['test_anno'] * self.output_spatial_scale).float()
        # boxes is in format xywh, convert it to x0y0x1y1 format
        boxes_72[:, 2:4] = boxes_72[:, 0:2] + boxes_72[:, 2:4]

        center_float = torch.stack(((boxes_72[:, 0] + boxes_72[:, 2]) / 2., (boxes_72[:, 1] + boxes_72[:, 3]) / 2.), dim=1)
        center_int = center_float.int().float()
        ind_72 = center_int[:, 1] * self.output_w + center_int[:, 0]  # [num_images, 1]

        data['ind_72'] = ind_72.long()
        data['w2h2_288'] = w2h2_288
        data['w2h2_72'] = w2h2_288 * 0.25

        ### Generate label functions
        if self.label_function_params is not None:
            # data['train_label'] = self._generate_label_function(data['train_anno'])
            data['test_label'] = self._generate_label_function(data['test_anno'])

            # data['train_label_36'] = self._generate_label_36_function(data['train_anno'])
            # data['test_label_36'] = self._generate_label_36_function(data['test_anno'])

            # data['train_label_72'] = self._generate_label_72_function(data['train_anno'])
            data['test_label_72'] = self._generate_label_72_function(data['test_anno'])

        return data


class SuperClsProcessing(BaseProcessing):

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, crop_type='replicate',
                 mode='pair', output_spatial_scale=1/4., output_w=72, output_h=72, max_scale_change=None,
                 proposal_params=None, label_function_params=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            crop_type - If 'replicate', the boundary pixels are replicated in case the search region crop goes out of image.
                        If 'nopad', the search region crop is shifted/shrunk to fit completely inside the image.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
            proposal_params - Arguments for the proposal generation process. See _generate_proposals for details.
            label_function_params - Arguments for the label generation process. See _generate_label_function for details.
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.crop_type = crop_type
        self.mode = mode

        self.label_function_params = label_function_params

        self.output_w = output_w
        self.output_h = output_h
        self.output_spatial_scale = output_spatial_scale

        self.max_scale_change = max_scale_change

        self.count = 0

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """
        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * self.center_jitter_factor[mode]).item()
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)


    def _generate_label_function(self, target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """
#####  ！！！！kernel_sz后面需要改一下，现在写死了为4
        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), self.label_function_params['sigma_factor'],
                                                      4,
                                                      self.label_function_params['feature_sz'], self.output_sz,
                                                      end_pad_if_even=self.label_function_params.get('end_pad_if_even', True))

        return gauss_label

    def _generate_label_72_function(self, target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """

        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), 1 / 40.,
                                                      self.label_function_params['kernel_sz'],
                                                      self.output_sz * self.output_spatial_scale, self.output_sz,
                                                      end_pad_if_even=self.label_function_params.get('end_pad_if_even',
                                                                                                     True))

        return gauss_label


    def _generate_label_36_function(self, target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """

        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), 1 / 30.,
                                                      self.label_function_params['kernel_sz'],
                                                      self.output_sz * self.output_spatial_scale / 2, self.output_sz,
                                                      end_pad_if_even=self.label_function_params.get('end_pad_if_even',
                                                                                                     True))

        return gauss_label

    def _create_dynamic_logisticloss_label(self, label_size, bbs, strides=16, rPos=2, rNeg=0):
        if isinstance(label_size, int):
            sz = label_size
        else:
            sz = label_size[0]

        # sz_x = sz // 2 + int(-c_shift[0] / 8)  # 8 is strides
        # sz_y = sz // 2 + int(-c_shift[1] / 8)
        labels = []
        for target_bb in bbs:
            target_center = (target_bb[0:2] + 0.5 * target_bb[2:4]) / strides
            x_c = target_center[0]
            y_c = target_center[1]

            # pytorch input is (NCHW) --> yx
            y, x = torch.meshgrid(torch.arange(0, sz) - torch.floor(y_c),
                                  torch.arange(0, sz) - torch.floor(x_c))

            dist_to_center = torch.abs(x) + torch.abs(y)  # Block metric
            label = torch.where(dist_to_center <= rPos,
                                 torch.ones_like(y),
                                 torch.where(dist_to_center < rNeg,
                                              0.5 * torch.ones_like(y),
                                              torch.zeros_like(y)))
            labels.append(label)
        labels = torch.stack(labels, dim=0) # num_images, H, W
        return labels


    def save_groundtruth_image(self, img, bbox, save_file = None):
        """Display a 2D tensor.
        args:
            fig_num: Figure number.
            title: Title of figure.
        """

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        draw_rec = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
        cv2.imwrite(save_file, draw_rec)

    def save_test_image(self, img,save_file = None):
        """Display a 2D tensor.
        args:
            fig_num: Figure number.
            title: Title of figure.
        """

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_file, img)

    def save_tensor_as_image(self, a, save_file=None, range=(None, None)):
        """Display a 2D tensor.
        args:
            fig_num: Figure number.
            title: Title of figure.
        """
        if a.ndim == 3:
            a = np.transpose(a, (1, 2, 0))
        plt.imsave(save_file, a)

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'train_images'  -
                'test_images'   -
                'train_anno'    -
                'test_anno'     -
        """

        # if self.transform['joint'] is not None:
        #     num_train_images = len(data['train_images'])
        #     all_images = data['train_images'] + data['test_images']
        #     all_images_trans = self.transform['joint'](*all_images)
        #
        #     data['train_images'] = all_images_trans[:num_train_images]
        #     data['test_images'] = all_images_trans[num_train_images:]

        # super dimp settings
        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'] = self.transform['joint'](image=data['train_images'],
                                                                               bbox=data['train_anno'])
            data['test_images'], data['test_anno'] = self.transform['joint'](image=data['test_images'],
                                                                             bbox=data['test_anno'], new_roll=False)

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # if self.crop_type == 'replicate':
            #     crops, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
            #                                                 self.search_area_factor, self.output_sz)
            # elif self.crop_type == 'nopad':
            #     crops, boxes = prutils.jittered_center_crop_nopad(data[s + '_images'], jittered_anno, data[s + '_anno'],
            #                                                       self.search_area_factor, self.output_sz)
            # else:
            #     raise ValueError('Unknown crop type {}'.format(self.crop_type))
            #
            # data[s + '_images'] = [self.transform[s](x) for x in crops]

            crops, boxes = prutils.target_image_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                     self.search_area_factor, self.output_sz, mode=self.crop_type,
                                                     max_scale_change=self.max_scale_change)

            data[s + '_images'], boxes = self.transform[s](image=crops, bbox=boxes, joint=False)

            boxes = torch.stack(boxes)
            boxes_init = boxes
            boxes_init[:, 2:4] = boxes[:, 0:2] + boxes[:, 2:4]
            boxes = boxes_init.clamp(0.0, self.output_sz - 1)

            boxes[:, 2:4] = boxes[:, 2:4] - boxes[:, 0:2]
            data[s + '_anno'] = boxes

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(prutils.stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        ### Generate label functions
        if self.label_function_params is not None:
            data['train_label'] = self._generate_label_function(data['train_anno'])
            data['test_label'] = self._generate_label_function(data['test_anno'])
            data['train_label_s'] = self._create_dynamic_logisticloss_label(18, data['train_anno'])
            data['test_label_s'] = self._create_dynamic_logisticloss_label(18, data['test_anno'])

        return data

class TransEncoderProcessing(BaseProcessing):

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, crop_type='replicate',
                 mode='pair', output_spatial_scale=1/4., output_w=72, output_h=72, max_scale_change=None,
                 proposal_params=None, label_function_params=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            crop_type - If 'replicate', the boundary pixels are replicated in case the search region crop goes out of image.
                        If 'nopad', the search region crop is shifted/shrunk to fit completely inside the image.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
            proposal_params - Arguments for the proposal generation process. See _generate_proposals for details.
            label_function_params - Arguments for the label generation process. See _generate_label_function for details.
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.crop_type = crop_type
        self.mode = mode

        self.label_function_params = label_function_params

        self.output_w = output_w
        self.output_h = output_h
        self.output_spatial_scale = output_spatial_scale

        self.max_scale_change = max_scale_change

        self.count = 0

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """
        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * self.center_jitter_factor[mode]).item()
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_label_function_s(self, target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """
        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4),
                                                      self.label_function_params['sigma_factor'],
                                                      3,
                                                      self.label_function_params['feature_sz'], self.output_sz,
                                                      end_pad_if_even=self.label_function_params.get(
                                                          'end_pad_if_even', True))

        return gauss_label


    def _generate_label_function(self, target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """
        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4),
                                                      self.label_function_params['sigma_factor'],
                                                      4,
                                                      self.label_function_params['feature_sz'], self.output_sz,
                                                      end_pad_if_even=self.label_function_params.get(
                                                          'end_pad_if_even', True))

        return gauss_label

    def _create_dynamic_logisticloss_label(self, label_size, bbs, strides=16, rPos=2, rNeg=0):
        if isinstance(label_size, int):
            sz = label_size
        else:
            sz = label_size[0]

        # sz_x = sz // 2 + int(-c_shift[0] / 8)  # 8 is strides
        # sz_y = sz // 2 + int(-c_shift[1] / 8)
        labels = []
        for target_bb in bbs:
            target_center = (target_bb[0:2] + 0.5 * target_bb[2:4]) / strides
            x_c = target_center[0]
            y_c = target_center[1]

            # pytorch input is (NCHW) --> yx
            y, x = torch.meshgrid(torch.arange(0, sz) - torch.floor(y_c),
                                  torch.arange(0, sz) - torch.floor(x_c))

            dist_to_center = torch.abs(x) + torch.abs(y)  # Block metric
            label = torch.where(dist_to_center <= rPos,
                                 torch.ones_like(y),
                                 torch.where(dist_to_center < rNeg,
                                              0.5 * torch.ones_like(y),
                                              torch.zeros_like(y)))
            labels.append(label)
        labels = torch.stack(labels, dim=0) # num_images, H, W
        return labels


    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'train_images'  -
                'test_images'   -
                'train_anno'    -
                'test_anno'     -
        """

        # if self.transform['joint'] is not None:
        #     num_train_images = len(data['train_images'])
        #     all_images = data['train_images'] + data['test_images']
        #     all_images_trans = self.transform['joint'](*all_images)
        #
        #     data['train_images'] = all_images_trans[:num_train_images]
        #     data['test_images'] = all_images_trans[num_train_images:]

        # super dimp settings
        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'] = self.transform['joint'](image=data['train_images'],
                                                                               bbox=data['train_anno'])
            data['test_images'], data['test_anno'] = self.transform['joint'](image=data['test_images'],
                                                                             bbox=data['test_anno'], new_roll=False)

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # Check whether data is valid. Avoid too small bounding boxes
            # w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]
            # crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor)
            # if (crop_sz < 1).any():
            #     data['valid'] = False
            #     print("Too small box is found. Replace it with new data.")
            #     return data

            # if self.crop_type == 'replicate':
            #     crops, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
            #                                                 self.search_area_factor, self.output_sz)
            # elif self.crop_type == 'nopad':
            #     crops, boxes = prutils.jittered_center_crop_nopad(data[s + '_images'], jittered_anno, data[s + '_anno'],
            #                                                       self.search_area_factor, self.output_sz)
            # else:
            #     raise ValueError('Unknown crop type {}'.format(self.crop_type))
            #
            # data[s + '_images'] = [self.transform[s](x) for x in crops]

            # crops, boxes = prutils.target_image_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
            #                                          self.search_area_factor, self.output_sz, mode=self.crop_type,
            #                                          max_scale_change=self.max_scale_change)

            crops, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                        self.search_area_factor, self.output_sz)

            data[s + '_images'], boxes = self.transform[s](image=crops, bbox=boxes, joint=False)

            boxes = torch.stack(boxes)
            boxes_init = boxes
            boxes_init[:, 2:4] = boxes[:, 0:2] + boxes[:, 2:4]
            boxes = boxes_init.clamp(0.0, self.output_sz - 1)

            # normalize
            boxes[:, 2:4] = boxes[:, 2:4] - boxes[:, 0:2]
            # boxes = boxes
            data[s + '_anno'] = boxes  # x0,y0,w,h (288)

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(prutils.stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        ### Generate label functions
        if self.label_function_params is not None:
            # data['train_label'] = self._generate_label_function(data['train_anno'])
            # data['test_label'] = self._generate_label_function(data['test_anno'])

            # data['train_label_s'] = self._generate_label_function_s(data['train_anno'])
            # data['test_label_s'] = self._generate_label_function_s(data['test_anno'])

            data['train_label_s'] = self._create_dynamic_logisticloss_label(18, data['train_anno'])
            data['test_label_s'] = self._create_dynamic_logisticloss_label(18, data['test_anno'])

        return data



class TransSiamProcessing(BaseProcessing):

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, crop_type='replicate',
                 mode='pair', output_spatial_scale=1/4., output_w=72, output_h=72, max_scale_change=None,
                 proposal_params=None, label_function_params=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            crop_type - If 'replicate', the boundary pixels are replicated in case the search region crop goes out of image.
                        If 'nopad', the search region crop is shifted/shrunk to fit completely inside the image.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
            proposal_params - Arguments for the proposal generation process. See _generate_proposals for details.
            label_function_params - Arguments for the label generation process. See _generate_label_function for details.
        """
        super().__init__(*args, **kwargs)
        self.crop_area_factor_sear = search_area_factor
        self.crop_area_factor_target = 2.0
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.crop_type = crop_type
        self.mode = mode

        self.label_function_params = label_function_params

        self.output_w = output_w
        self.output_h = output_h
        self.output_spatial_scale = output_spatial_scale

        self.max_scale_change = max_scale_change

        self.count = 0

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """
        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * self.center_jitter_factor[mode]).item()
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_label_function_s(self, target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """
        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4),
                                                      self.label_function_params['sigma_factor'],
                                                      3,
                                                      self.label_function_params['feature_sz'], self.output_sz,
                                                      end_pad_if_even=self.label_function_params.get(
                                                          'end_pad_if_even', True))

        return gauss_label


    def _generate_label_function(self, target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """
        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4),
                                                      self.label_function_params['sigma_factor'],
                                                      4,
                                                      self.label_function_params['feature_sz'], self.output_sz,
                                                      end_pad_if_even=self.label_function_params.get(
                                                          'end_pad_if_even', True))

        return gauss_label

    def _create_dynamic_logisticloss_label(self, label_size, bbs, strides=16, rPos=2, rNeg=0):
        if isinstance(label_size, int):
            sz = label_size
        else:
            sz = label_size[0]

        # sz_x = sz // 2 + int(-c_shift[0] / 8)  # 8 is strides
        # sz_y = sz // 2 + int(-c_shift[1] / 8)
        labels = []
        for target_bb in bbs:
            target_center = (target_bb[0:2] + 0.5 * target_bb[2:4]) / strides
            x_c = target_center[0]
            y_c = target_center[1]

            # pytorch input is (NCHW) --> yx
            y, x = torch.meshgrid(torch.arange(0, sz) - torch.floor(y_c),
                                  torch.arange(0, sz) - torch.floor(x_c))

            dist_to_center = torch.abs(x) + torch.abs(y)  # Block metric
            label = torch.where(dist_to_center <= rPos,
                                 torch.ones_like(y),
                                 torch.where(dist_to_center < rNeg,
                                              0.5 * torch.ones_like(y),
                                              torch.zeros_like(y)))
            labels.append(label)
        labels = torch.stack(labels, dim=0) # num_images, H, W
        return labels


    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'train_images'  -
                'test_images'   -
                'train_anno'    -
                'test_anno'     -
        """

        # super dimp settings
        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'] = self.transform['joint'](image=data['train_images'],
                                                                               bbox=data['train_anno'])
            data['test_images'], data['test_anno'] = self.transform['joint'](image=data['test_images'],
                                                                             bbox=data['test_anno'], new_roll=False)

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            if s == 'train':
                self.search_area_factor = self.crop_area_factor_target
                self.output_sz = 128
            else:
                self.search_area_factor = self.crop_area_factor_sear
                self.output_sz = 288
            crops, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                        self.search_area_factor, self.output_sz)

            data[s + '_images'], boxes = self.transform[s](image=crops, bbox=boxes, joint=False)

            boxes = torch.stack(boxes)
            boxes_init = boxes
            boxes_init[:, 2:4] = boxes[:, 0:2] + boxes[:, 2:4]
            boxes = boxes_init.clamp(0.0, self.output_sz - 1)

            # normalize
            boxes[:, 2:4] = boxes[:, 2:4] - boxes[:, 0:2]
            # boxes = boxes
            data[s + '_anno'] = boxes  # x0,y0,w,h (288)

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(prutils.stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        ### Generate label functions
        if self.label_function_params is not None:
            # data['train_label'] = self._generate_label_function(data['train_anno'])
            # data['test_label'] = self._generate_label_function(data['test_anno'])

            # data['train_label_s'] = self._generate_label_function_s(data['train_anno'])
            # data['test_label_s'] = self._generate_label_function_s(data['test_anno'])

            # data['train_label_s'] = self._create_dynamic_logisticloss_label(18, data['train_anno'])
            data['test_label_s'] = self._create_dynamic_logisticloss_label(18, data['test_anno'])

        return data



class TransClsProcessing(BaseProcessing):

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, crop_type='replicate',
                 mode='pair', output_spatial_scale=1/4., output_w=72, output_h=72, max_scale_change=None,
                 proposal_params=None, label_function_params=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            crop_type - If 'replicate', the boundary pixels are replicated in case the search region crop goes out of image.
                        If 'nopad', the search region crop is shifted/shrunk to fit completely inside the image.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
            proposal_params - Arguments for the proposal generation process. See _generate_proposals for details.
            label_function_params - Arguments for the label generation process. See _generate_label_function for details.
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.crop_type = crop_type
        self.mode = mode

        self.label_function_params = label_function_params

        self.output_w = output_w
        self.output_h = output_h
        self.output_spatial_scale = output_spatial_scale

        self.max_scale_change = max_scale_change

        self.count = 0

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """
        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * self.center_jitter_factor[mode]).item()
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_label_function_s(self, target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """
        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4),
                                                      self.label_function_params['sigma_factor'],
                                                      3,
                                                      self.label_function_params['feature_sz'], self.output_sz,
                                                      end_pad_if_even=self.label_function_params.get(
                                                          'end_pad_if_even', True))

        return gauss_label


    def _generate_label_function(self, target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """
        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4),
                                                      self.label_function_params['sigma_factor'],
                                                      4,
                                                      self.label_function_params['feature_sz'], self.output_sz,
                                                      end_pad_if_even=self.label_function_params.get(
                                                          'end_pad_if_even', True))

        return gauss_label


    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'train_images'  -
                'test_images'   -
                'train_anno'    -
                'test_anno'     -
        """

        # if self.transform['joint'] is not None:
        #     num_train_images = len(data['train_images'])
        #     all_images = data['train_images'] + data['test_images']
        #     all_images_trans = self.transform['joint'](*all_images)
        #
        #     data['train_images'] = all_images_trans[:num_train_images]
        #     data['test_images'] = all_images_trans[num_train_images:]

        # super dimp settings
        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'] = self.transform['joint'](image=data['train_images'],
                                                                               bbox=data['train_anno'])
            data['test_images'], data['test_anno'] = self.transform['joint'](image=data['test_images'],
                                                                             bbox=data['test_anno'], new_roll=False)

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # Check whether data is valid. Avoid too small bounding boxes
            # w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]
            # crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor)
            # if (crop_sz < 1).any():
            #     data['valid'] = False
            #     print("Too small box is found. Replace it with new data.")
            #     return data

            # if self.crop_type == 'replicate':
            #     crops, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
            #                                                 self.search_area_factor, self.output_sz)
            # elif self.crop_type == 'nopad':
            #     crops, boxes = prutils.jittered_center_crop_nopad(data[s + '_images'], jittered_anno, data[s + '_anno'],
            #                                                       self.search_area_factor, self.output_sz)
            # else:
            #     raise ValueError('Unknown crop type {}'.format(self.crop_type))
            #
            # data[s + '_images'] = [self.transform[s](x) for x in crops]

            crops, boxes = prutils.target_image_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                     self.search_area_factor, self.output_sz, mode=self.crop_type,
                                                     max_scale_change=self.max_scale_change)

            # crops, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
            #                                             self.search_area_factor, self.output_sz)

            data[s + '_images'], boxes = self.transform[s](image=crops, bbox=boxes, joint=False)

            boxes = torch.stack(boxes)
            boxes_init = boxes
            boxes_init[:, 2:4] = boxes[:, 0:2] + boxes[:, 2:4]
            boxes = boxes_init.clamp(0.0, self.output_sz - 1)

            # normalize
            boxes[:, 2:4] = boxes[:, 2:4] - boxes[:, 0:2]
            # boxes = boxes
            data[s + '_anno'] = boxes  # x0,y0,w,h (288)

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(prutils.stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        ### Generate label functions
        if self.label_function_params is not None:
            data['train_label'] = self._generate_label_function(data['train_anno'])
            data['test_label'] = self._generate_label_function(data['test_anno'])

            data['train_label_s'] = self._generate_label_function_s(data['train_anno'])
            data['test_label_s'] = self._generate_label_function_s(data['test_anno'])



        return data
