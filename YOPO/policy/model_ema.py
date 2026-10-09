"""
Model EMA (Exponential Moving Average), following YOLOv5's ModelEMA.
Keeps a moving average of the weights, used for validation and checkpointing.
"""
import math
from copy import deepcopy

import torch


class ModelEMA:
    def __init__(self, model, decay=0.999, tau=1000):
        """decay: upper bound of the decay.
        tau: ramp-up speed, d = decay * (1 - exp(-step/tau)). Lower it when the dataset is small.
        """
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.tau = tau
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        model_state = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(model_state[k].detach(), alpha=1 - d)
            else:
                v.copy_(model_state[k])  # integer buffers, e.g. BN's num_batches_tracked
