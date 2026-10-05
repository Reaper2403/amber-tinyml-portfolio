"""Audio model definitions extracted from the original training notebook builder."""
from torch import nn

MODEL_CFG = {'stem_filters': 32, 'block_filters': [32, 48, 64, 64], 'dropout': 0.15}

class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=(1, 1), dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.net(x)

class AudioDSCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        stem_filters = MODEL_CFG["stem_filters"]
        blocks = MODEL_CFG["block_filters"]
        dropout = MODEL_CFG["dropout"]

        self.stem = nn.Sequential(
            nn.Conv2d(1, stem_filters, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(stem_filters),
            nn.ReLU(inplace=True),
        )

        layers = []
        in_channels = stem_filters
        for idx, out_channels in enumerate(blocks):
            stride = (2, 2) if idx in (1, 3) else (1, 1)
            layers.append(
                DepthwiseSeparableBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
        self.blocks = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.20),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)
