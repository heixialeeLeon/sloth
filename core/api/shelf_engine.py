from abc import abstractclassmethod, ABCMeta
from typing import Dict, Tuple, Sequence, List
from core.api.offtake_detector import OfftakeInfo

class ShelfEngine(metaclass=ABCMeta):
    '''
    the main class for the shelf related function
    '''

    def diff_areas(self, img) -> OfftakeInfo:
        '''
        get the offtake infos
        :param img:
        :return:
        '''