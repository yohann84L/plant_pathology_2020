import torch
import torchvision


class PlantModel(torch.nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool = True, finetune: bool = True, num_classes: int = 4):
        super().__init__()
        self.model_name = backbone_name
        self.backbone = self.build_backbone(backbone_name, pretrained, finetune, num_classes)
        self.logits = torch.nn.ModuleList(
            [torch.nn.Linear(self.backbone.num_ftrs, c) for c in range(num_classes)]
        )

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x = self.backbone(x)

        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        x = torch.nn.functional.dropout(x, 0.25, self.training)

        logits = [logit(x) for logit in self.logits]

        return logits

    def build_backbone(self, base_model_name: str, pretrained: bool, finetune: bool, num_classes: int):
        base_model_accepted = [
            "mobilenetv2",
            "vgg16",
            "resnet18",
            "resnet50",
            "resnext50"
        ]

        # Mobilenet v2
        if base_model_name == "mobilenetv2":
            backbone = torchvision.models.mobilenet_v2(pretrained).features
            backbone.out_channels = 1280
        # VGG 16
        elif base_model_name == "vgg16":
            backbone = torchvision.models.vgg16(pretrained).features
            if finetune:
                self.set_grad_for_finetunning(backbone, 10)
            backbone.out_channels = 512
        # ResNet 18
        elif base_model_name == "resnet18":
            backbone = torchvision.models.resnet18(pretrained)
            num_ftrs = backbone.fc.in_features
            backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
            #print(backbone)
            if finetune:
                self.set_grad_for_finetunning(backbone, 7)
            #backbone.fc = torch.nn.Linear(num_ftrs, num_classes)
            backbone.num_ftrs = num_ftrs


        # ResNet 50
        elif base_model_name == "resnet50":
            backbone = torch.nn.Sequential(*list(torchvision.models.resnet50(pretrained).children())[:-2])
            if finetune:
                self.set_grad_for_finetunning(backbone, 7)


            #backbone.fc = torch.nn.Linear(num_ftrs, num_classes)
        # ResNext 50
        elif base_model_name == "resnext50":
            backbone = torch.nn.Sequential(*list(torchvision.models.resnext50_32x4d(pretrained).children())[:-2])
            if finetune:
                self.set_grad_for_finetunning(backbone, 7)
            backbone.out_channels = 2048
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
