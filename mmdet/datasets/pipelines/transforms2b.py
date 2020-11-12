import inspect

import mmcv
import numpy as np
from numpy import random

from mmdet.core import PolygonMasks
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from ..builder import PIPELINES

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None


@PIPELINES.register_module()
class CutImage(object):
    def __init__(self):
        pass

    def __call__(self, results):
        img = results["img"]
        h, w = img.shape[:2]
        img_l = img[:, :w//2]
        img_r = img[:, w//2:]
        results["img_shape"] = img_l.shape
        results["ori_shape"] = img_l.shape
        results["img"] = img_l
        results["img2"] = img_r
        results["img_fields"].append("img2")
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + '()'
        return repr_str
