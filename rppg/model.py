from torch import nn

from . import NO_PULSE


class ConvNet3D(nn.Module):
    """Process (batch, channel, height, width, time) video patches."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(20, 20, 58)),
            nn.MaxPool3d(2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(288, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, NO_PULSE + 1),
        )

    def forward(self, inputs):
        return self.fc(self.layer1(inputs))
