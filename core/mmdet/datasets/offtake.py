from .builder import DATASETS
from .coco import CocoDataset

@DATASETS.register_module()
class OfftakeDataset(CocoDataset):

    CLASSES = ('object')
