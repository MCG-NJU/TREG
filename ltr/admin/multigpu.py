import torch.nn as nn


def is_multi_gpu(net):
    return isinstance(net, (MultiGPU, nn.DataParallel, nn.parallel.DistributedDataParallel))


class MultiGPU(nn.DataParallel):
    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except:
            pass
        return getattr(self.module, item)