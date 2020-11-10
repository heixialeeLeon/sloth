from abc import abstractclassmethod, ABCMeta
from typing import Dict, Tuple, Sequence, List
import json

class OfftakeArea(object):
    def __init__(self, type: int, location: List):
        self.type = type
        self.location = location

class OfftakeInfo(object):
    def __init__(self):
        self.diff = list()

    def add_area(self, area: OfftakeArea):
        '''
        add the diff area record
        :param area:
        :return:
        '''
        self.diff.append(area)

    def to_json(self):
        return json.dumps(self, default=lambda o:o.__dict__, sort_keys=False)

class OfftakeDetector(metaclass=ABCMeta):

    def process(self, img) -> OfftakeInfo:
        '''
        detect the different areas for the left and right img
        :param img: the img with left and right
        :return: the different areas 
        '''
        pass