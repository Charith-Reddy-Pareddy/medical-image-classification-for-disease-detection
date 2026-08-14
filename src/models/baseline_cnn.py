from torch import nn


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class BaselineCNN(nn.Module):
    """From-scratch CNN, kept as a lower-bound benchmark against the
    transfer-learning models, not a novelty claim.
    """

    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
