import argparse
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import PlantPathologyDataset, DatasetTransforms
from models import PlantModel
from models.engine import train_one_epoch, evaluate
from optimizer import RAdam
from utils.metric_logger import *
from utils.utils import str2bool, get_device, save_checkpoint

from losses.focal_loss import FocalLoss


def parse_args():
    """
    Parse input arguments
    Returns:
        args
    """
    parser = argparse.ArgumentParser(description="Train Faster R-CNN model on TrayVisor dataset")
    parser.add_argument("--img_root", dest="img_root",
                        help="training dataset", type=str)
    parser.add_argument("--annot_train", dest="annot_train",
                        help="annotation train df", type=str)
    parser.add_argument("--annot_test", dest="annot_test",
                        help="annotation test df", type=str)
    parser.add_argument("--sample_sub", dest="sample_sub",
                        help="sample submission", type=str)
    parser.add_argument("--backbone", dest="backbone",
                        help="backbone name", type=str,
                        default="mobilenetv2")
    parser.add_argument("--pretrained", dest="pretrained",
                        help="use pretrained model", type=str2bool, nargs="?",
                        default=True)
    parser.add_argument("--num_classes", dest="num_classes",
                        help="number of classes", type=int,
                        default=4)
    parser.add_argument("--epochs", dest="epochs",
                        help="number of epochs", type=int,
                        default=10)
    parser.add_argument("--use_cuda", dest="use_cuda",
                        help="use cuda or not", type=str2bool, nargs="?",
                        default=True)
    parser.add_argument("--checkpoints", dest="checkpoints",
                        help="epoch at which you want to save the model. If -1 save only last epoch.",
                        nargs='+', type=int,
                        default=-1)
    parser.add_argument("--checkpoints_dir", dest="checkpoints_dir",
                        help="save checkpoints directory", type=str)
    parser.add_argument("--img_size", dest="img_size",
                        help="resize img to the given int", type=int,
                        nargs=2, default=-1)
    parser.add_argument("--batch_size", dest="batch_size",
                        help="batch size", type=int,
                        default=8)
    args = parser.parse_args()
    return args


def train(model, optimizer, criterion, lr_scheduler, data_loader: DataLoader, data_loader_test: DataLoader,
          num_epochs: int = 10, use_cuda: bool = True,
          epoch_save_ckpt: Union[int, list] = None, dir: str = None):
    """
    Method to train FasterRCNN_SaladFruit model.
    Args:
        data_loader (torch.utils.data.DataLoader): data loader to train model on
        data_loader_test (torch.utils.data.DataLoader): data loader to evaluate model on
        num_epochs (int = 10): number of epoch to train model
        use_cuda (bool = True): use cuda or not
        epoch_save_ckpt (list or int): Epoch at which you want to save the model. If -1 save only last epoch.
        dir (str = "models/): Directory where model are saved under the name "{model_name}_{date}_ep{epoch}.pth"
    """
    if epoch_save_ckpt == -1:
        epoch_save_ckpt = [num_epochs - 1]
    if not dir:
        dir = "checkpoints"
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    # choose device
    device = get_device(use_cuda)
    print(f"Using device {device.type}")
    # define dataset
    model.to(device)
    writer = SummaryWriter("logs")
    metric_logger_train = MetricLogger(delimiter="  ")
    # writer_test = SummaryWriter("runs/test")
    # metric_logger_test = MetricLogger(delimiter="  ", writer=writer_test)

    for epoch in metric_logger_train.log_every(range(num_epochs), print_freq=1, epoch=0, header="Training"):
        # train for one epoch, printing every 50 iterations
        train_metric = train_one_epoch(model, optimizer, data_loader, criterion, device, epoch, print_freq=40)
        # metric_logger_train.update(**train_metric)

        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        test_metric = evaluate(model, criterion, data_loader_test, device=device)
        # metric_logger_test.update(**test_metric)
        for key in train_metric.keys():
            writer.add_scalars(
                "metrics/{}".format(key), {
                    "{}_train".format(key, key): train_metric[key],
                    "{}_test".format(key, key): test_metric[key],
                }, global_step=epoch
            )
        # save checkpoint
        if epoch in epoch_save_ckpt:
            save_checkpoint(model, optimizer, dir.as_posix(), epoch)
    writer.close()
    # writer_train.close()
    # writer_test.close()

    print("That's it!")


if __name__ == '__main__':
    args = parse_args()
    dataset = PlantPathologyDataset(annot_fp=args.annot_train, img_root=args.img_root,
                                    transforms=DatasetTransforms(train=True, img_size=args.img_size))
    dataset_test = PlantPathologyDataset(annot_fp=args.annot_train, img_root=args.img_root,
                                         transforms=DatasetTransforms(train=False, img_size=args.img_size))

    model = PlantModel(
        backbone_name=args.backbone,
        pretrained=args.pretrained,
        num_classes=args.num_classes
    )

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    size = int(len(indices) * 0.15)
    #dataset = torch.utils.data.Subset(dataset, indices[:-size])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-size:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = RAdam(
        params=params,
        lr=1e-3,
        weight_decay=0.0005
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=3,
        gamma=0.5
    )

    # df_annots = dataset.dataset.annots
    # count = df_annots[["healthy", "multiple_diseases", "rust", "scab"]].sum().values
    # count = torch.from_numpy(count.astype(np.float32))
    # weight_classes = torch.tensor([1]) - (count / len(dataset))
    # weight_classes = weight_classes.to(get_device(args.use_cuda))
    # print(weight_classes)

    #criterion = FocalLoss(alpha=1.0, reduction="mean")
    criterion = torch.nn.BCEWithLogitsLoss()

    print("Start training")
    train(model, optimizer, criterion, lr_scheduler, data_loader, data_loader_test, num_epochs=args.epochs,
          use_cuda=args.use_cuda,
          epoch_save_ckpt=args.checkpoints, dir=args.checkpoints_dir)
