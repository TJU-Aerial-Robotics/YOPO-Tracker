import torch.nn as nn


class YopoHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(YopoHead, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_dim, 256, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(256, output_dim, kernel_size=1, stride=1)
        )

    def forward(self, x):
        return self.model(x)
