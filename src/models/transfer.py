from torch import nn
from torchvision import models


class ResNet50Transfer(nn.Module):
    """ImageNet-pretrained ResNet-50, fine-tuned end to end. Standard
    transfer-learning comparison against the from-scratch baseline.
    """

    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class DenseNet121Transfer(nn.Module):
    """ImageNet-pretrained DenseNet-121. Dense feature reuse is well suited
    to the subtle, diffuse texture patterns pneumonia opacities present as.
    """

    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.densenet121(weights=weights)
        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
