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

import torch
from .autoaugment import ImageNetPolicy

_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}

class DatasetTransformsAutoAug(object):
    def __init__(self, train=True, img_size=None, cutout=False, lighting=False):
        self.transforms = []
        if img_size is None or img_size == -1:
            img_size = (224, 224)
        self.img_size = img_size
        print(self.img_size)

        # Add base transform
        if train:
            self.add_train_transforms(cutout, lighting)
        else:
            self.add_test_transforms()
        # Add normalization
        self.add_normalization()

    def __call__(self, x):
        return Compose(self.transforms)(x)

    def add_train_transforms(self, cutout, lighting):
        self.transforms += [
            Resize(self.img_size),
            RandomHorizontalFlip(),
            ImageNetPolicy(),
            torchvision.transforms.ToTensor(),
        ]
        if cutout:
            self.transforms += [
                RandomErasing(p=0.4)
            ]
        if lighting:
            self.transforms += [
                Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
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


class UnNormalize(object):
    def __init__(self, mean: iter = None, std: iter = None):
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


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

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))