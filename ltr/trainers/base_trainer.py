import os
import glob
import torch
import traceback
from ltr.admin import loading, multigpu
from torch.utils.data.distributed import DistributedSampler
import time


class BaseTrainer:
    """Base trainer class. Contains functions for training and saving/loading chackpoints.
    Trainer classes should inherit from this one and overload the train_epoch function."""

    def __init__(self, actor, loaders, optimizer, settings, basic_device, lr_scheduler=None, logging_file=None,
                 max_norm=0.0):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        self.actor = actor
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loaders = loaders
        self.logging_file = logging_file

        self.update_settings(settings)

        self.epoch = 0
        self.max_norm = max_norm
        self.stats = {}

        self.device = basic_device
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() and settings.use_gpu else "cpu")

        self.actor.to(self.device)

    def update_settings(self, settings=None):
        """Updates the trainer settings. Must be called to update internal settings."""
        if settings is not None:
            self.settings = settings

        if self.settings.env.workspace_dir is not None:
            self.settings.env.workspace_dir = os.path.expanduser(self.settings.env.workspace_dir)
            self._checkpoint_dir = os.path.join(self.settings.env.workspace_dir, 'checkpoints')
            if self.settings.local_rank in [-1, 0]:
                if not os.path.exists(self._checkpoint_dir):
                    os.makedirs(self._checkpoint_dir)
        else:
            self._checkpoint_dir = None


    def train(self, max_epochs, load_latest=False, fail_safe=True):
        """Do training for the given number of epochs.
        args:
            max_epochs - Max number of training epochs,
            load_latest - Bool indicating whether to resume from latest epoch.
            fail_safe - Bool indicating whether the training to automatically restart in case of any crashes.
        """

        epoch = -1
        num_tries = 10
        for i in range(num_tries):
            try:
                if load_latest:
                    # if self.settings.local_rank in [-1, 0]:
                    self.load_checkpoint()
                for epoch in range(self.epoch+1, max_epochs+1):
                    self.epoch = epoch
                    self.train_epoch()

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    if self._checkpoint_dir:
                        ### used for DDP
                        if self.settings.local_rank in [-1, 0]:
                            self.save_checkpoint()

            except:
                # raise Exception("Error.")
                print('Training crashed at epoch {}'.format(epoch))
                if fail_safe:
                    self.epoch -= 1
                    load_latest = True
                    print('Traceback for the error!')
                    print(traceback.format_exc())
                    print('Restarting training from last epoch ...')
                else:
                    raise

        print('Finished training!')


    def train_epoch(self):
        raise NotImplementedError

    def eval_otb_epoch(self, threads=0, epoch=None, thre=None, search_area=None, train_skipping=None, reg_net_opt_iter=None,
                       net_opt_update_iter=None, lamda=None, distractor_threshold=None, reg_update_iter=None, reg_lamda=None,
                       reg_init_train_iter=None, net_opt_iter=None, init_train_iter=None, init_train_frames=None, iou_select=None, merge_rate=None):
        raise NotImplementedError


    def save_checkpoint(self):
        """Saves a checkpoint of the network and other variables."""

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__
        state = {
            'epoch': self.epoch,
            'actor_type': actor_type,
            'net_type': net_type,
            'net': net.state_dict(),
            'net_info': getattr(net, 'info', None),
            'constructor': getattr(net, 'constructor', None),
            'optimizer': self.optimizer.state_dict(),
            'stats': self.stats,
            'settings': self.settings
        }


        directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = '{}/{}_ep{:04d}.pth.tar'.format(directory, net_type, self.epoch)
        torch.save(state, file_path)


    def load_checkpoint(self, checkpoint = None, fields = None, ignore_fields = None, load_constructor = False):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__

        if checkpoint is None:
            # Load most recent checkpoint
            # print(self._checkpoint_dir, self.settings.project_path, net_type)
            checkpoint_list = sorted(glob.glob('{}/{}/{}_ep*.pth.tar'.format(self._checkpoint_dir,
                                                                             self.settings.project_path, net_type)))

            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                print('No matching checkpoint file found')
                return
        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_path = '{}/{}/{}_ep{:04d}.pth.tar'.format(self._checkpoint_dir, self.settings.project_path,
                                                                 net_type, checkpoint)
        elif isinstance(checkpoint, str):
            # checkpoint is the path
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(glob.glob('{}/*_ep*.pth.tar'.format(checkpoint)))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception('No checkpoint found')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        # Load network
        # checkpoint_dict = loading.torch_load_legacy(checkpoint_path)
        # t1 = time.time()
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
        # print("load model to cpu cost: {}".format(time.time() - t1))

        assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'

        if fields is None:
            fields = checkpoint_dict.keys()
        if ignore_fields is None:
            ignore_fields = ['settings']

            # Never load the scheduler. It exists in older checkpoints.
        ignore_fields.extend(['lr_scheduler', 'constructor', 'net_type', 'actor_type', 'net_info'])

        # Load all fields
        for key in fields:
            if key in ignore_fields:
                continue
            if key == 'net':
                # t2 = time.time()
                net.load_state_dict(checkpoint_dict[key])
                # print("load weights to model cost: {}".format(time.time() - t2))
                # dict = net.state_dict()
                # load_fcotnet_dict = {'.'.join(k.split('.')[1:]): v for k, v in checkpoint_dict['net'].items()}
                # dict.update(load_fcotnet_dict)
                # net.load_state_dict(dict)
            elif key == 'optimizer':
                self.optimizer.load_state_dict(checkpoint_dict[key])
            else:
                setattr(self, key, checkpoint_dict[key])

        # Set the net info
        if load_constructor and 'constructor' in checkpoint_dict and checkpoint_dict['constructor'] is not None:
            net.constructor = checkpoint_dict['constructor']
        if 'net_info' in checkpoint_dict and checkpoint_dict['net_info'] is not None:
            net.info = checkpoint_dict['net_info']

        # # Update the epoch in lr scheduler
        # if 'epoch' in fields:
        #     self.lr_scheduler.last_epoch = self.epoch

        # Update the epoch in lr scheduler
        if 'epoch' in fields:
            self.lr_scheduler.last_epoch = self.epoch
            # Update the epoch in data_samplers
            for loader in self.loaders:
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)

        return True
