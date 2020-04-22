# ----------------------------------------
# Author: Yohann Lereclus
# Description: Faster RCNN implementation
# ----------------------------------------

import torch
from torch.utils.data.dataloader import DataLoader

from losses.mixup import MixupCriterion
from models.metrics import ComputeMetrics
from models.mixup import MixupData
from utils.metric_logger import MetricLogger, SmoothedValue


def train_one_epoch(model, optimizer: torch.optim, data_loader: DataLoader, criterion: torch.nn.modules.loss,
                    device: torch.device, epoch: int, print_freq: int):
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
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    metric = ComputeMetrics(n_classes=4)

    running_loss, epoch_loss = 0, 0

    mixup_data = MixupData()
    mixup_criterion = MixupCriterion()

    # Iterate over dataloader to train model
    for images, labels in metric_logger.log_every(data_loader, print_freq, epoch, header):
        images = images.to(device)
        labels = labels.to(device)

        mixup = False
        # Compute loss
        if mixup:
            inputs, targets_a, targets_b, lam = mixup_data(images, labels)
            inputs, targets_a, targets_b = map(torch.tensor, (inputs, targets_a, targets_b))

            outputs = model(inputs)

            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        running_loss += loss.item()

        # Process backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute ROC AUC
        metric.update(outputs, labels)

        # metrics = compute_roc_auc(outputs, labels)
        metric_logger.update(loss=loss)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    epoch_metric = metric.get_auc_roc()
    epoch_metric["loss"] = running_loss / len(data_loader.dataset)

    print(epoch_metric)
    return epoch_metric


@torch.no_grad()
def evaluate(model, criterion: torch.nn.modules.loss, data_loader: DataLoader, device: torch.device):
    """
    Method to evaluate Plant model on a test set.
    Args:
        model (FasterRCNN): model to evaluate
        data_loader (torch.utils.data.DataLoader): data loader to test model on
        device (torch.device): device to use, either device("cpu") or device("cuda")
        criterion (loss): loss
        writer (SummaryWriter = None): set a writer if you want to write log file in tensorboards
        epoch (int = None): state epoch of the current training if there is one
    """
    model.eval()

    metric = ComputeMetrics(n_classes=4)

    running_loss, epoch_loss = 0, 0

    # Iterate over dataloader to train model
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)

        # Compute ROC AUC
        metric.update(outputs, labels)

    epoch_metric = metric.get_auc_roc()
    epoch_metric["loss"] = running_loss / len(data_loader.dataset)

    return epoch_metric
