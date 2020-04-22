from statistics import mean

import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
from tabulate import tabulate

class ComputeMetrics(object):
    def __init__(self, n_classes=4):
        self.n_classes = n_classes
        self.output = None
        self.target = None

    def update(self, output: torch.tensor, target: torch.tensor):
        if (self.output is None) or (self.target is None):
            self.output = output
            self.target = target
        else:
            self.output = torch.cat((self.output, output))
            self.target = torch.cat((self.target, target))

    def get_auc_roc(self) -> dict:
        roc_auc, fpr, tpr = {}, {}, {}
        output = torch.nn.functional.softmax(self.output, dim=1)
        output = output.cpu().detach().numpy()
        target = self.target.cpu().detach().numpy().astype(np.uint8)

        for i in range(self.n_classes):
            key = "class{:d}".format(i)
            fpr[key], tpr[key], _ = roc_curve(target[:, i], output[:, i])
            roc_auc[key] = auc(fpr[key], tpr[key])
        mean_auc = mean(roc_auc.values())
        roc_auc["mean_auc"] = mean_auc
        return roc_auc


def print_result_table(train_metric: dict, test_metric: dict):
    headers = ["metric", "train", "test", "diff"]
    data = sorted(
        [(k, v_train, v_test, v_train/v_test)
         for (k, v_train), (_, v_test)
         in zip(train_metric.items(), test_metric.items())]
    )
    print(tabulate(data, headers=headers))