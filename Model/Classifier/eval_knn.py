import os

import torch
import torch.nn as nn
import torch.distributed as dist

from torchvision import datasets
from torchvision import transforms


class KnnModule(nn.Module):
    def __init__(self):
        super(KnnModule, self).__init__()