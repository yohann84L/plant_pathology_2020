import torch
import torchvision
from efficientnet_pytorch import EfficientNet

from models.efficientnet import EfficientNet as CustomEfficientNet


class PlantModel(torch.nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool = True, finetune: bool = True, num_classes: int = 4):
        super().__init__()
        self.model_name = backbone_name
        self.backbone = self.build_backbone(backbone_name, pretrained, finetune, num_classes)

    def forward(self, x):
        return self.backbone(x)

    def build_backbone(self, base_model_name: str, pretrained: bool, finetune: bool, num_classes: int):
        base_model_accepted = [
            "mobilenetv2",
            "vgg16",
            "resnet18",
            "resnet50",
            "resnext50"
            "efficientnetb2",
            "efficientnetb5",
        ]

        efficientnetns = ["efficientnetnsb" + str(i) for i in range(1, 8)]
        base_model_accepted += efficientnetns

        # Mobilenet v2
        if base_model_name == "mobilenetv2":
            backbone = torchvision.models.mobilenet_v2(pretrained).features
            if finetune:
                self.set_grad_for_finetunning(backbone, 7)
            num_ftrs = backbone.classifier[-1].in_features
            backbone.classifier[-1] = torch.nn.Linear(num_ftrs, num_classes)
        # VGG 16
        elif base_model_name == "vgg16":
            backbone = torchvision.models.vgg16(pretrained).features
            if finetune:
                self.set_grad_for_finetunning(backbone, 10)
            num_ftrs = backbone.classifier[-1].in_features
            backbone.classifier[-1] = torch.nn.Linear(num_ftrs, num_classes)
        # ResNet 18
        elif base_model_name == "resnet18":
            backbone = torchvision.models.resnet18(pretrained)
            if finetune:
                self.set_grad_for_finetunning(backbone, 7)
            num_ftrs = backbone.fc.in_features
            backbone.fc = torch.nn.Linear(num_ftrs, num_classes)
        # ResNet 50
        elif base_model_name == "resnet50":
            backbone = torchvision.models.resnet50(pretrained)
            if finetune:
                self.set_grad_for_finetunning(backbone, 7)
            num_ftrs = backbone.fc.in_features
            backbone.fc = torch.nn.Linear(num_ftrs, num_classes)
        # ResNext 50
        elif base_model_name == "resnext50":
            backbone = torchvision.models.resnext50_32x4d(pretrained)
            if finetune:
                self.set_grad_for_finetunning(backbone, 7)
            num_ftrs = backbone.fc.in_features
            backbone.fc = torch.nn.Linear(num_ftrs, num_classes)
        # EfficientNet b2
        elif base_model_name == "efficientnetb2":
            backbone = EfficientNet.from_pretrained("efficientnet-b2")
            if finetune:
                self.set_grad_for_finetunning(backbone, 2)
            num_ftrs = backbone._fc.in_features
            backbone._fc = torch.nn.Linear(num_ftrs, num_classes)
        # EfficientNet b4
        elif base_model_name == "efficientnetb4":
            backbone = EfficientNet.from_pretrained("efficientnet-b4")
            if finetune:
                self.set_grad_for_finetunning(backbone, 4)
            num_ftrs = backbone._fc.in_features
            backbone._fc = torch.nn.Linear(num_ftrs, num_classes)
        # EfficientNet b5
        elif base_model_name == "efficientnetb5":
            backbone = EfficientNet.from_pretrained("efficientnet-b5")
            if finetune:
                self.set_grad_for_finetunning(backbone, 3)
            num_ftrs = backbone._fc.in_features
            backbone._fc = torch.nn.Linear(num_ftrs, num_classes)
        # EfficientNet b5
        elif base_model_name == "efficientnetb7":
            backbone = EfficientNet.from_pretrained("efficientnet-b7")
            if finetune:
                self.set_grad_for_finetunning(backbone, 3)
            num_ftrs = backbone._fc.in_features
            backbone._fc = torch.nn.Linear(num_ftrs, num_classes)
        # EfficientNet Noisy Student
        elif base_model_name[:-1] == "efficientnetnsb":
            n = base_model_name[-1]
            backbone = CustomEfficientNet.from_custom_pretrained("efficientnet-ns-b" + str(n), dir="checkpoints/")
            if finetune:
                self.set_grad_for_finetunning(backbone, 3)
            num_ftrs = backbone._fc.in_features
            backbone._fc = torch.nn.Linear(num_ftrs, num_classes)
        else:
            print("Backbone model should be one of the following list: ")
            for name in base_model_accepted:
                print("     - {}".format(name))
            raise NotImplementedError
        return backbone

    @staticmethod
    def set_grad_for_finetunning(backbone, layer_number):
        count = 0
        for child in backbone.children():
            count += 1
            if count < layer_number:
                for param in child.parameters():
                    param.requires_grad = False
