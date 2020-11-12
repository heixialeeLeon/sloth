from abc import abstractclassmethod, ABCMeta
from typing import Dict, Tuple, Sequence, List
from core.api.offtake_detector import OfftakeInfo

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
        adjust_img = self.shelf_adjust.process(img)
        return self.offtake_det.process(adjust_img)
