# ----------------------------------------
# Author: Yohann Lereclus
# Description: Faster RCNN implementation
# ----------------------------------------

import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor
from cv2 import BORDER_REFLECT


class DatasetTransforms:
    def __init__(self, train=True, img_size=None):
        self.train = train
        self.transforms = []
        if img_size is not None:
            img_size = (224, 224)
        self.img_size = img_size

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
                A.Resize(650, 650),
                A.RandomCrop(600, 600),
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
                A.Resize(600, 600),
            ]

    def add_normalization(self):
        self.transforms += [
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ]