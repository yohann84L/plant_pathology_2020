# ----------------------------------------
# Author: Yohann Lereclus
# Description: Faster RCNN implementation
# ----------------------------------------

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

classes = ("healthy", "multiple_diseases", "rust", "scab")


class PlantPathologyDataset(Dataset):
    """
    Plant Pathology custom dataset.
    Arguments:
        annot_fp (str): filepath for the annotation file
        img_root (str): path for the image folder
        transforms: transforms
    """

    def __init__(self, annot_fp: str, img_root: str, transforms=None):
        self.annots = pd.read_csv(Path(annot_fp).as_posix())
        self.transforms = transforms
        self.img_root = Path(img_root)

    def __getitem__(self, idx):
        obj = self.annots.iloc[idx]

        # Get image
        img_path = Path(self.img_root) / Path(obj["image_id"]).with_suffix(".jpg")

        # Get image and transforms it if necessary for augmentation
        # img = cv2.imread(img_path.as_posix())
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # if self.transforms:
        #     data = {"image": img}
        #     res = self.transforms(**data)
        #     img = res["image"]

        img = Image.open(img_path.as_posix())

        if self.transforms:
            img = self.transforms(img)

        # Convert each numpy array into tensor
        labels = torch.tensor(obj.values[1:].tolist(), dtype=torch.float32)

        return img, labels

    def __len__(self):
        return self.annots.shape[0]
