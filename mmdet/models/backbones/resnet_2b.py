import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import kaiming_init
from .resnet import ResNet
from ..builder import BACKBONES


@BACKBONES.register_module()
class ResNet2b(nn.Module):
    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True):
        super(ResNet2b, self).__init__()
        self.backbone = ResNet(depth, in_channels, stem_channels, base_channels, num_stages, strides, dilations,
                            out_indices, style, deep_stem, avg_down, frozen_stages, conv_cfg, norm_cfg, norm_eval,
                            dcn, stage_with_dcn, plugins, with_cp, zero_init_residual)
        self.convs = nn.ModuleList([nn.Conv2d(c * 2, c, 1) for c in [256, 512, 1024, 2048]])

    def forward(self, x1, x2):
        outs1 = self.backbone(x1)
        outs2 = self.backbone(x2)
        outs = []
        for o1, o2, conv in zip(outs1, outs2, self.convs):
            outs.append(conv(torch.cat([o1, o2], 1)))
        return tuple(outs)

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained)
        for m in self.convs:
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
