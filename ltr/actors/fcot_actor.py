from . import BaseActor
from ltr.models.loss.target_regression import DenseIouPred

class FcotActor(BaseActor):
    """Actor for jointly training cls_18, cls_72 branch and regression branch, except for regression optimizer"""
    def __init__(self, net, objective, device, loss_weight=None):
        super().__init__(net, objective, device)
        if loss_weight is None:
            loss_weight = {'test_clf_72': 100, 'test_init_clf_72': 100, 'test_iter_clf_72': 400,
                           'test_clf_18': 100, 'test_init_clf_18': 100, 'test_iter_clf_18': 400,
                           'reg_72': 1}
        self.loss_weight = loss_weight
        self.dense_iou_pred = DenseIouPred(dim=4)

    def __call__(self, data, gen_iou_map=False, vis=False):
        """
        args:
            data - The input data.
            gen_iou_map -
        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores_18, target_scores_72, offset_maps = self.net(train_imgs=data['train_images'],
                                                                   test_imgs=data['test_images'],
                                                                   train_bb=data['train_anno'])
        # target_scores_18, target_scores_72, offset_maps = output['target_scores_18'], output['target_scores_72'], output['offset_maps']

        ### Regression offline loss ###
        reg_loss_72 = self.objective['reg_72'](offset_maps, data['ind_72'], data['w2h2_72'], radius=2) * self.loss_weight['reg_72']

        # reg_loss_72 = [self.objective['reg_72'](offset_map, data['ind_72'], data['w2h2_72'], radius=2) * self.loss_weight['reg_72'] for offset_map in offset_maps]

        # if gen_iou_map:
        #     pred_iou_map = self.dense_iou_pred(offset_maps, data['ind_72'], data['w2h2_72'], radius=5)     # [h, w]
        # else:
        #     pred_iou_map = None
        # pred_iou_map = None

        ### Classification-72 losses ###
        clf_losses_test_72 = [self.objective['test_clf_72'](s, data['test_label_72'], data['test_anno']) for s in target_scores_72]

        # Loss of the final filter
        clf_loss_test_72 = clf_losses_test_72[-1]
        loss_target_classifier_72 = self.loss_weight['test_clf_72'] * clf_loss_test_72

        # Loss for the initial filter iteration
        loss_test_init_clf_72 = 0
        if 'test_init_clf_72' in self.loss_weight.keys():
            loss_test_init_clf_72 = self.loss_weight['test_init_clf_72'] * clf_losses_test_72[0]

        # Loss for the intermediate filter iterations
        loss_test_iter_clf_72 = 0
        if 'test_iter_clf_72' in self.loss_weight.keys():
            test_iter_weights_72 = self.loss_weight['test_iter_clf_72']
            if isinstance(test_iter_weights_72, list):
                loss_test_iter_clf_72 = sum([a*b for a, b in zip(test_iter_weights_72, clf_losses_test_72[1:-1])])
            else:
                loss_test_iter_clf_72 = (test_iter_weights_72 / (len(clf_losses_test_72) - 2)) * sum(clf_losses_test_72[1:-1])


        ### Classification-18 losses ###
        clf_losses_test_18 = [self.objective['test_clf_18'](s, data['test_label'], data['test_anno']) for s in
                              target_scores_18]

        # Loss of the final filter
        clf_loss_test_18 = clf_losses_test_18[-1]
        loss_target_classifier_18 = self.loss_weight['test_clf_18'] * clf_loss_test_18

        # Loss for the initial filter iteration
        loss_test_init_clf_18 = 0
        if 'test_init_clf_18' in self.loss_weight.keys():
            loss_test_init_clf_18 = self.loss_weight['test_init_clf_18'] * clf_losses_test_18[0]

        # Loss for the intermediate filter iterations
        loss_test_iter_clf_18 = 0
        if 'test_iter_clf_18' in self.loss_weight.keys():
            test_iter_weights_18 = self.loss_weight['test_iter_clf_18']
            if isinstance(test_iter_weights_18, list):
                loss_test_iter_clf_18 = sum(
                    [a * b for a, b in zip(test_iter_weights_18, clf_losses_test_18[1:-1])])
            else:
                loss_test_iter_clf_18 = (test_iter_weights_18 / (len(clf_losses_test_18) - 2)) * sum(
                    clf_losses_test_18[1:-1])

        # Total loss
        loss = loss_target_classifier_72 + loss_test_init_clf_72 + loss_test_iter_clf_72 + \
               reg_loss_72 + \
              loss_target_classifier_18 + loss_test_init_clf_18 + loss_test_iter_clf_18


        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/reg_72': reg_loss_72.item(),
                 'Loss/target_clf_72': loss_target_classifier_72.item(),
                 'Loss/target_clf_18': loss_target_classifier_18.item(),
                }
        if 'test_init_clf_72' in self.loss_weight.keys():
            stats['Loss/test_init_clf_72'] = loss_test_init_clf_72.item()
        if 'test_iter_clf_72' in self.loss_weight.keys():
            stats['Loss/test_iter_clf_72'] = loss_test_iter_clf_72.item()
        stats['ClfTrain/test_loss_72'] = clf_loss_test_72.item()
        if len(clf_losses_test_72) > 0:
            stats['ClfTrain/test_init_loss'] = clf_losses_test_72[0].item()
            if len(clf_losses_test_72) > 2:
                stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test_72[1:-1]).item() / (len(clf_losses_test_72) - 2)

        if 'test_init_clf_18' in self.loss_weight.keys():
            stats['Loss/test_init_clf_18'] = loss_test_init_clf_18.item()
        if 'test_iter_clf_18' in self.loss_weight.keys():
            stats['Loss/test_iter_clf_18'] = loss_test_iter_clf_18.item()

        vis_results = {}

        return loss, stats, vis_results