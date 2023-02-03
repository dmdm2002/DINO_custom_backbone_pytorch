import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.Backbone.VIG.PyramidModules import Stem, Downsample
from Model.Backbone.VIGA.ViGAModules import ViG_Block

import torchsummary


class Pyramid_ViGA(nn.Module):
    def __init__(self, num_class=1, k=9, supervision=False):
        super(Pyramid_ViGA, self).__init__()
        self.num_class = num_class
        self.supervision = supervision
        _k = k
        _repeat = [2, 2, 16, 2]
        _dimensions = [96, 192, 384, 768]
        self.stem = Stem(in_channel=3, out_channel=_dimensions[0])

        vig_1 = [ViG_Block(in_channel=_dimensions[0], k=_k, dilation=1) for _ in range(_repeat[0])]

        self.vig_1 = nn.Sequential(*vig_1)
        self.down_1 = Downsample(in_channel=_dimensions[0], out_channel=_dimensions[1])

        vig_2 = [ViG_Block(in_channel=_dimensions[1], k=_k, dilation=1) for _ in range(_repeat[1])]

        self.vig_2 = nn.Sequential(*vig_2)
        self.down_2 = Downsample(in_channel=_dimensions[1], out_channel=_dimensions[2])

        vig_3 = [ViG_Block(in_channel=_dimensions[2], k=_k, dilation=1) for _ in range(_repeat[2])]

        self.vig_3 = nn.Sequential(*vig_3)
        self.down_3 = Downsample(in_channel=_dimensions[2], out_channel=_dimensions[3])

        vig_4 = [ViG_Block(in_channel=_dimensions[3], k=_k, dilation=1) for _ in range(_repeat[3])]

        self.vig_4 = nn.Sequential(*vig_4)

        if self.supervision:
            self.output1 = nn.Conv2d(_dimensions[192], _dimensions[192])

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(_dimensions[3]*1, self.num_class)

    def forward(self, x):
        x = self.stem(x)

        x1 = self.vig_1(x)
        x1 = self.down_1(x1)

        x2 = self.vig_2(x1)
        x2 = self.down_2(x2)

        x3 = self.vig_3(x2)
        x3 = self.down_3(x3)

        x4 = self.vig_4(x3)

        x_pool = self.pool(x4)
        x_view = x_pool.view(x.size(0), -1)
        output = self.fc(x_view)

        return output
