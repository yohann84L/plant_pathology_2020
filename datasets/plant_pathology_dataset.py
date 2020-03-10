# ----------------------------------------
# Author: Yohann Lereclus
# Description: Faster RCNN implementation
# ----------------------------------------

from pathlib import Path
from typing import Union

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms
from torch.utils.data.dataset import Dataset


class Classes:
    label_str_to_id = {
        "background": 0,
        "small_vrac": 1,
        "big_vrac": 2
    }

    id_to_label_str = {
        0: "background",
        1: "small_vrac",
        2: "big_vrac"
    }


class PlantPathologyDataset(Dataset):
    """
    Plant Pathology custom dataset.
    Arguments:
        annot_fp (str): filepath for the annotation file
        img_root (str): path for the image folder
        transforms: transforms
    """

    def __init__(self, annot_fp: str, img_root: str, transforms: torchvision.transforms.Compose = None):
        self.annots = pd.read_csv(Path(annot_fp).as_posix())
        self.transforms = transforms
        self.img_root = Path(img_root)

    def __getitem__(self, idx):
        obj = self.annots.iloc[idx]

        # Get image
        img_path = Path(self.img_root)/Path(obj["image_id"]).with_suffix(".jpg")

        # Get image and transforms it if necessary for augmentation
        img = cv2.imread(img_path.as_posix())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            self.transforms(img)

        # Convert each numpy array into tensor
        labels = obj.values[1:]

        return img, labels

    def __len__(self):
        return self.annots.shape[0]
