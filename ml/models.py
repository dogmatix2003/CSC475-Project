"""Models: MLP, CNN and CRNN. All output 25 scores per frame (24 chords + N)."""
import torch.nn as nn

from ml import config


class MLP(nn.Module):
    """Simple classifier on the flattened CQT patch."""

    def __init__(self, context=config.CONTEXT, width=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.CQT_BINS * (2 * context + 1), width), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width, width // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width // 2, config.N_CLASSES))

    def forward(self, x):# (B, 1, 144, 2c+1) -> (B, 25)
        return self.net(x)


def conv_block(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU())


class CNN(nn.Module):
    """Small convolutional network on the CQT patch."""

    def __init__(self, context=config.CONTEXT, width=16, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, width), conv_block(width, width), nn.MaxPool2d((2, 1)),# 144 -> 72 bins
            conv_block(width, 2 * width), conv_block(2 * width, 2 * width), nn.MaxPool2d((2, 1)),# -> 36
            conv_block(2 * width, 4 * width), nn.MaxPool2d((3, 1)),# -> 12
            nn.AdaptiveAvgPool2d((12, 1)), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(4 * width * 12, 256), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(256, config.N_CLASSES))

    def forward(self, x):# (B, 1, 144, 2c+1) -> (B, 25)
        return self.head(self.features(x))


class CRNN(nn.Module):
    """Convolutions + bidirectional GRU over a chunk; the GRU can learn smoothing by itself."""

    def __init__(self, context=None, width=16, hidden=64):
        super().__init__()
        self.conv = nn.Sequential(
            conv_block(1, width), nn.MaxPool2d((2, 1)),
            conv_block(width, width), nn.MaxPool2d((2, 1)),
            conv_block(width, 2 * width), nn.MaxPool2d((3, 1)))# 144 -> 12 bins, every frame kept
        self.gru = nn.GRU(2 * width * 12, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, config.N_CLASSES)

    def forward(self, x):# (B, 1, 144, T) -> (B, 25, T)
        h = self.conv(x)
        B, C, F, T = h.shape
        h, _ = self.gru(h.permute(0, 3, 1, 2).reshape(B, T, C * F))
        return self.out(h).permute(0, 2, 1)


MODELS = {"mlp": MLP, "cnn": CNN, "crnn": CRNN}
DEFAULT_WIDTH = {"mlp": 512, "cnn": 16, "crnn": 16}


def build_model(name, context=config.CONTEXT, width=None):
    return MODELS[name](context=context, width=width or DEFAULT_WIDTH[name])
