# ----------------------------------------
# Author: Yohann Lereclus
# Description: Faster RCNN implementation
# ----------------------------------------

from statistics import mean

import torch
from sklearn.metrics import roc_curve, auc
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.metric_logger import MetricLogger, SmoothedValue


def train_one_epoch(model, optimizer: torch.optim, data_loader: DataLoader, criterion: torch.nn.modules.loss,
                    device: torch.device, epoch: int, print_freq: int, writer: SummaryWriter):
    """
    Method to train Plant model one time
    Args:
        model (plant_model): model to train
        optimizer (torch.optimizer): optimizer used
        criterion (loss): loss
        data_loader (torch.utils.data.DataLoader): data loader to test model on
        device (torch.device): device to use, either device("cpu") or device("cuda")
        epoch (int = None): state epoch of the current training if there is one
        print_freq (int = None): print frequency of log writer
        writer (SummaryWriter = None): set a writer if you want to write log file in tensorboard
    """
    # Set model to train mode
    model.train()
    # Define metric logger parameters
    metric_logger = MetricLogger(delimiter="  ", writer=writer)
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    # Iterate over dataloader to train model
    for images, labels in metric_logger.log_every(data_loader, print_freq, epoch, header):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        losses = []
        print(outputs)
        print(labels)
        for i in range(4):
            losses.append(criterion(outputs[i], labels[:, i]))
        loss = sum(losses)

        # Process backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics = compute_roc_auc(outputs, labels)
        metric_logger.update(loss=loss, **metrics)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


@torch.no_grad()
def evaluate(model, data_loader: DataLoader, device: torch.device, criterion: torch.nn.modules.loss,
             print_freq: int, writer: SummaryWriter = None, epoch: int = None):
    """
    Method to evaluate FasterRCNN_SaladFruit model on a test set.
    Args:
        model (FasterRCNN): model to evaluate
        data_loader (torch.utils.data.DataLoader): data loader to test model on
        device (torch.device): device to use, either device("cpu") or device("cuda")
        criterion (loss): loss
        writer (SummaryWriter = None): set a writer if you want to write log file in tensorboards
        epoch (int = None): state epoch of the current training if there is one
    """
    model.eval()
    # Define metric logger parameters
    metric_logger = MetricLogger(delimiter="  ", writer=writer)
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Eval: [{}]'.format(epoch)

    model_time_list = []
    total_loss = 0

    for images, labels in metric_logger.log_every(data_loader, print_freq, epoch, header):
        images = list(image.to(device) for image in images)
        labels = list(label.to(device) for label in labels)

        outputs = model(images, labels)

        losses = []
        for i in range(4):
            losses.append(criterion(outputs[i], labels[:, i]))
        loss = sum(losses)
        total_loss += loss

        metrics = compute_roc_auc(outputs, labels)
        metric_logger.update(loss=loss, **metrics)


def compute_roc_auc(output, target, n_classes=4) -> dict:
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[f"classe {i}"], tpr[f"classe {i}"], _ = roc_curve(output[:, i], target[:, i])
        roc_auc[f"classe {i}"] = auc(fpr[f"classe {i}"], tpr[f"classe {i}"])
    mean_auc = mean(roc_auc.values)
    roc_auc["mean_auc"] = mean_auc
    return roc_auc
