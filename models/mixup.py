import numpy as np
import torch

from utils import get_device


class MixupData(object):
    def __init__(self, alpha: float = 1.0, use_cuda: bool = True):
        self.alpha = alpha
        self.device = get_device(use_cuda)

    def __call__(self, x, y):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam


class MixupCriterion(object):
    def __call__(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
