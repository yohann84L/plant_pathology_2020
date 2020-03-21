# ----------------------------------------
# Author: Yohann Lereclus
# Description: Faster RCNN implementation
# ----------------------------------------

import albumentations as A
import torchvision
from albumentations.pytorch import ToTensorV2 as ToTensor
from cv2 import BORDER_REFLECT
from torchvision.transforms import (
    Resize,
    RandomHorizontalFlip,
    RandomErasing,
    Normalize,
    Compose,
)

from .autoaugment import ImageNetPolicy


class DatasetTransformsAutoAug(object):
    def __init__(self, train=True, img_size=None, cutout=False):
        self.transforms = []
        if img_size is None or img_size == -1:
            img_size = (224, 224)
        self.img_size = img_size
        print(self.img_size)

        # Add base transform
        if train:
            self.add_train_transforms(cutout)
        else:
            self.add_test_transforms()
        # Add normalization
        self.add_normalization()

    def __call__(self, x):
        return Compose(self.transforms)(x)

    def add_train_transforms(self, cutout):
        self.transforms += [
            Resize(self.img_size),
            RandomHorizontalFlip(),
            ImageNetPolicy(),
            torchvision.transforms.ToTensor()
        ]
        if cutout:
            self.transforms += [
                RandomErasing(p=0.4)
            ]

    def add_test_transforms(self):
        self.transforms += [
            Resize(self.img_size),
            torchvision.transforms.ToTensor()
        ]

    def add_normalization(self):
        self.transforms += [
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]


class DatasetTransformsAlbumentation:
    def __init__(self, train=True, img_size=None):
        self.train = train
        self.transforms = []
        if img_size is None or img_size == -1:
            img_size = (224, 224)
        self.img_size = img_size
        print(self.img_size)

        # Add base tranform
        self.add_transforms()
        # Add normalization
        self.add_normalization()
        # Convert with ToTensor()
        self.transforms.append(ToTensor())

    def __call__(self, **data):
        return A.Compose(self.transforms)(**data)

    def add_transforms(self):
        if self.train:
            self.transforms += [
                A.Resize(int(self.img_size[0] * 1.1), int(self.img_size[1] * 1.1)),
                A.RandomCrop(self.img_size[0], self.img_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(p=0.5, border_mode=BORDER_REFLECT, value=0),

                # Pixels
                A.OneOf([
                    A.IAAEmboss(p=1.0),
                    A.IAASharpen(p=1.0),
                    A.Blur(p=1.0),
                ], p=0.5),

                # Affine
                A.OneOf([
                    A.ElasticTransform(p=1.0),
                    A.IAAPiecewiseAffine(p=1.0)
                ], p=0.5),
            ]
        else:
            self.transforms += [
                A.Resize(self.img_size[0], self.img_size[1]),
            ]

    def add_normalization(self):
        self.transforms += [
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ]
