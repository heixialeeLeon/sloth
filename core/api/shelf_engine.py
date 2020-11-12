from abc import abstractclassmethod, ABCMeta
from typing import Dict, Tuple, Sequence, List
from core.api.offtake_detector import OfftakeInfo
from core.api.shelf_adjust import ShelfSegmentation
from core.api.offtake_detector import OfftakeDetector

class ShelfEngine(metaclass=ABCMeta):
    '''
    the main class for the shelf related function
    '''
    def __init__(self, shelf_adjust, offtake_det):
        self.shelf_adjust = shelf_adjust
        self.offtake_det = offtake_det

    def diff_areas(self, img) -> OfftakeInfo:
        '''
        get the offtake infos
        :param img:
        :return:
        '''
        adjust_img = self.shelf_adjust(img)
        return self.offtake_det(adjust_img)
