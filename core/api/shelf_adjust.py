from abc import abstractclassmethod, ABCMeta
from typing import Dict, Tuple, Sequence, List
import numpy as np

class ShelfAdjust(metaclass=ABCMeta):

    def process(self, img):
        '''
        Process the img and align the left and the right
        :param img: the img with left and right
        :return:  the aligned img with left and right
        '''
        pass