import torch
import torch.nn as nn
import copy

from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad


class DINO_model(nn.Module):
    def __init__(self, backbone, input_dim):
        super(DINO_model, self).__init__()
        self.student_backbone = backbone
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(input_dim, 768, 96, 2048, freeze_last_layer=1)
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 768, 96, 2048)

        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z