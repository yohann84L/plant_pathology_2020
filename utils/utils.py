import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist

from models import PlantModel


def save_checkpoint(model, optimizer, dir: str, epoch: int):
    """
    Save a model checkpoint at a given epoch.
    Args:
        dir: dir folder to save the .pth file
        epoch: epoch the model is
    """
    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'model_name': model.model_name}
    now = datetime.now()
    filename = "{model_name}_{date}_ep{epoch}.pth".format(
        model_name=model.model_name,
        date=now.strftime("%b%d_%H-%M"),
        epoch=epoch
    )
    check_dir(dir)
    torch.save(state, Path(dir) / filename)
    "Checkpoint saved : {}".format(Path(dir) / filename)


def load_for_inference(filename: str, cuda: bool = True) -> PlantModel:
    """
    Load a model checkpoint to make inference.
    Args:
        filename (str): filename/path of the checkpoint.pth
        cuda (bool = True): use cuda
    Returns:
        (FasterRCNNFood) model
    """
    device = torch.device("cuda") if (cuda and torch.cuda.is_available()) else torch.device("cpu")
    if Path(filename).exists():
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location=device)
        # Load params
        model_name = checkpoint['model_name']
        # Build model key/architecture
        model = PlantModel(model_name, pretrained=True, num_classes=4)
        # Update model and optimizer
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model = model.eval()

        print("=> loaded checkpoint '{}'".format(filename))
        return model
    else:
        print("=> no checkpoint found at '{}'".format(filename))


def load_for_training(filename: str, cuda: bool = True) -> PlantModel:
    """
    Load a model checkpoint to make inference.
    Args:
        filename (str): filename/path of the checkpoint.pth
        cuda (bool = True): use cuda
    Returns:
        (FasterRCNNFood) model
    """
    device = torch.device("cuda") if (cuda and torch.cuda.is_available()) else torch.device("cpu")
    if Path(filename).exists():
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location=device)
        # Load params
        model_name = checkpoint['model_name']
        # Build model key/architecture
        model = PlantModel(model_name, pretrained=True, num_classes=4)
        # Update model and optimizer
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        optimizer = checkpoint['optimizer']
        epoch = checkpoint['epoch']

        print("=> loaded checkpoint '{}'".format(filename))
        return model
    else:
        print("=> no checkpoint found at '{}'".format(filename))


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_device(use_cuda: bool) -> torch.device:
    if use_cuda:
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("/!\ Warning : You set use_cuda however no GPU seems to be available. CPU is set intead.")
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def collate_fn(batch):
    return tuple(zip(*batch))


def check_dir(dir: str):
    Path(dir).mkdir(parents=True, exist_ok=True)
