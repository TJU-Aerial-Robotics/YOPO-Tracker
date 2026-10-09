import time
import torch
import torch.nn
from torch.hub import load_state_dict_from_url
from config.config import cfg
from policy.models.resnet import resnet18, model_urls


def _init_from_imagenet(cnn):
    """Load the torchvision ImageNet weights into cnn; fc and output_layer are not covered.
    Returns conv1's 3-channel weight, for the caller to inflate before replacing conv1.
    """
    report = cnn.load_state_dict(load_state_dict_from_url(model_urls['resnet18'], progress=False),
                                 strict=False)
    n = len([k for k in cnn.state_dict() if not k.startswith("output_layer")])
    n_missing = len([k for k in report.missing_keys if not k.startswith("output_layer")])
    print(f"[backbone] ImageNet init: {n - n_missing}/{n} tensors loaded"
          + (f", still random: {[k for k in report.missing_keys if not k.startswith('output_layer')]}"
             if n_missing else ""))
    return cnn.conv1.weight.data.clone()


# input: [1, 4, 192, 320]
class ResNet18(torch.nn.Module):
    def __init__(self, output_dim: int, pretrained: bool = False):
        super(ResNet18, self).__init__()
        self.cnn = resnet18(pretrained=False)
        conv1_rgb = _init_from_imagenet(self.cnn) if pretrained else None
        self.cnn.conv1 = torch.nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if conv1_rgb is not None:
            with torch.no_grad():  # RGB keeps the pretrained filters, depth starts from their mean
                self.cnn.conv1.weight[:, :3] = conv1_rgb
                self.cnn.conv1.weight[:, 3:] = conv1_rgb.mean(dim=1, keepdim=True)
        self.cnn.output_layer = torch.nn.Conv2d(512, output_dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.cnn(depth)


# Faster and smaller (input: [1, 32, 64])
class ResNet14(torch.nn.Module):
    def __init__(self, output_dim: int):
        super(ResNet14, self).__init__()
        self.cnn = resnet18(pretrained=False)
        self.cnn.conv1 = torch.nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.layer4 = torch.nn.Sequential()
        self.cnn.output_layer = torch.nn.Conv2d(256, output_dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.cnn(depth)


def YopoBackbone(output_dim):
    return ResNet18(output_dim, pretrained=cfg["train"] and bool(cfg["pretrained_backbone"]))


if __name__ == '__main__':
    net = YopoBackbone(64)
    input_ = torch.zeros((1, 4, 192, 320))
    start = time.time()
    output = net(input_)
    print(time.time() - start)
